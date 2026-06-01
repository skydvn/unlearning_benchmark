import os
import time
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from approx_algo.gradient_ascent import Gradient_Ascent


class Module2(Gradient_Ascent):
    """
    ModULE: localized, expert-level (un)learning on a per-layer MoE backbone.

    This edition fixes three deviations from the paper:
      (1) At-risk retained set D_r^risk: retain / distill / separation losses are
          applied ONLY to retained samples whose active expert set intersects the
          selected forget-relevant experts M_f (Eq. risk_retain_set). Invariant
          samples have frozen forward paths and contribute nothing, so they are
          masked out and the surviving losses are averaged over |D_r^risk|.
      (2) Routing regularizers (sparsity, balance) read the DENSE routing
          distribution pi = softmax(router_logits) over all M experts
          (`last_pi_all`), not the top-k slot weights (`last_pi`).
      (3) The optimizer is scoped to the CHOSEN experts only (M_f, by the
          forget-specific responsibility score rho_m = s_m(D_f) - alpha * s_m(D_r)).
          Selection happens once per request (as in the paper) and the optimizer
          is built over exactly those parameters.

    NOTE: requires `DeepMoELayer.forward` to cache `last_topk_indices` of shape
    (B, S, gate_k) so the at-risk set can be computed. See the accompanying
    architecture patch.
    """

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        unseen_loader,
        forget_loader,
        forget_test_loader,
        retain_loader,
        retain_test_loader,
        optimizer,
        criteria,
        num_epoch,
        # config for learn
        lambda_sparse=1.0,
        lambda_balance=1.0,
        lambda_div=1.0,
        # config for unlearn
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        eta=1.0,
        k_u=2,
        device="cuda",
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            unseen_loader=unseen_loader,
            forget_loader=forget_loader,
            forget_test_loader=forget_test_loader,
            retain_loader=retain_loader,
            retain_test_loader=retain_test_loader,
            optimizer=optimizer,
            criteria=criteria,
            num_epoch=num_epoch,
            device=device,
        )

        actual_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        supported_models = ["ModuleArchitecture"]
        if actual_model.__class__.__name__ not in supported_models:
            raise TypeError(
                f"Module does not support {self.model.__class__.__name__}. "
                f"Supported: {supported_models}"
            )

        self.lambda_sparse = lambda_sparse
        self.lambda_balance = lambda_balance
        self.lambda_div = lambda_div

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.k_u = k_u

    # ==========================================
    # helper methods for learn phase
    # ==========================================
    def _loss_sparse(self, pi):
        # instance-level routing entropy over ALL M experts (dense pi).
        entropy = -(pi * (pi + 1e-8).log()).sum(dim=-1)
        return entropy.mean()

    def _loss_balance(self, pi, module_name, use_ema, ema_states, ema_alpha):
        # dataset-level anti-collapse: pull mean usage of each of the M experts
        # toward 1/M. `pi` must be the dense (token, M) distribution.
        M = pi.size(-1)
        mean_pi = pi.mean(dim=0)

        if use_ema:
            if module_name not in ema_states:
                ema_states[module_name] = torch.ones_like(mean_pi) / M
            effective_pi = ema_alpha * ema_states[module_name] + (1 - ema_alpha) * mean_pi
            ema_states[module_name] = effective_pi.detach()
        else:
            effective_pi = mean_pi

        return ((effective_pi - 1.0 / M) ** 2).sum()

    def _loss_diversity(self, h_stack, eps=1e-6):
        # cross-expert decorrelation on raw expert outputs.
        B, M, r = h_stack.shape
        if M < 2:
            return h_stack.new_zeros(())

        loss = h_stack.new_zeros(())
        H_tilde = []
        for m in range(M):
            H_m = h_stack[:, m, :]
            norm_F = H_m.norm(p="fro").clamp(min=eps)
            H_tilde.append(H_m / norm_F)

        for m in range(M):
            for n in range(M):
                if m == n:
                    continue
                C = (H_tilde[m].T @ H_tilde[n]) / B
                loss = loss + (C ** 2).sum()
        return loss

    # ==========================================
    # helper methods for unlearn phase
    # ==========================================
    def _get_routing_mass(self, loader):
        # average DENSE routing mass per expert over a dataset (Eq. forget/retain
        # routing mass). Uses last_pi_all (softmax over all M experts).
        self.model.eval()
        masses = None
        total_tokens = 0
        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(self.device)
                self.model(images)

                batch_masses = []
                num_tokens = 0
                for _, m in self.model.named_modules():
                    if m.__class__.__name__ == "DeepMoELayer":
                        batch_masses.append(m.last_pi_all.sum(dim=0))
                        num_tokens = m.last_pi_all.size(0)

                if masses is None:
                    masses = batch_masses
                else:
                    masses = [a + b for a, b in zip(masses, batch_masses)]
                total_tokens += num_tokens

        return [m / max(total_tokens, 1) for m in masses]

    def _select_forget_experts(self, moe_layers):
        """
        Fix (3): forget-specific responsibility selection.
        rho_m = s_m(D_f) - alpha * s_m(D_r); M_f = TopK_{k_u}(rho_m).
        Done once per request. Returns the per-layer selected expert lists and,
        as a side effect, freezes everything except the chosen experts.
        """
        forget_mass = self._get_routing_mass(self.forget_loader)
        retain_mass = self._get_routing_mass(self.retain_loader)

        # encoder, routers, head, and ALL experts -> start from unlearning mode
        # (backbone + routers + head frozen, experts trainable), then narrow to M_f.
        self.model._set_grad_mode("unlearning")

        selected_experts_per_layer = []
        for l_idx, m in enumerate(moe_layers):
            rho_m = forget_mass[l_idx] - self.alpha * retain_mass[l_idx]
            _, topk_indices = rho_m.topk(self.k_u, dim=-1)
            selected = topk_indices.tolist()
            selected_experts_per_layer.append(selected)

            for expert_idx, expert in enumerate(m.experts):
                is_selected = expert_idx in selected
                for param in expert.parameters():
                    param.requires_grad = is_selected

        return selected_experts_per_layer

    @torch.no_grad()
    def _compute_risk_mask(self, moe_layers, selected_experts_per_layer):
        """
        Fix (1): build the at-risk sample mask for the CURRENT cached forward.
        A retained sample is at-risk if, in ANY layer, ANY of its tokens routes
        (top-k) to a selected forget-expert. Returns a (B,) boolean tensor.

        Relies on `m.last_topk_indices` of shape (B, S, gate_k) being populated by
        the most recent forward pass (here: the retain batch).
        """
        risk = None
        for l_idx, m in enumerate(moe_layers):
            idx = m.last_topk_indices  # (B, S, k)
            sel = torch.as_tensor(
                selected_experts_per_layer[l_idx], device=idx.device, dtype=idx.dtype
            )
            # (B, S, k, |sel|) -> any over experts-in-sel -> (B, S, k)
            hits = (idx.unsqueeze(-1) == sel.view(1, 1, 1, -1)).any(dim=-1)
            sample_hit = hits.flatten(1).any(dim=1)  # (B,)
            risk = sample_hit if risk is None else (risk | sample_hit)
        return risk

    def _unlearn_loss_forget(self, logits_f, labels_f):
        # gradient-ascent forgetting on D_f.
        return -self.criteria(logits_f, labels_f)

    def _unlearn_loss_retain(self, logits_r_risk, labels_r_risk):
        # retained-task preservation on D_r^risk only.
        return self.criteria(logits_r_risk, labels_r_risk)

    def _unlearn_loss_distill(self, logits_r_risk, images_r_risk, origin_model):
        # output-level stability on D_r^risk: KL(p_old || p_new). Teacher in eval.
        with torch.no_grad():
            orig_logits, _ = origin_model.inference(images_r_risk)
        log_p_new = F.log_softmax(logits_r_risk, dim=-1)
        p_old = F.softmax(orig_logits, dim=-1)
        return F.kl_div(log_p_new, p_old, reduction="batchmean")

    def _unlearn_loss_separation(self, moe_layers, selected_experts_per_layer, token_risk_mask):
        # separation between updated and frozen experts, on at-risk retained tokens.
        loss_sep = torch.zeros((), device=self.device)
        for l_idx, m in enumerate(moe_layers):
            H = m.last_h  # (B*S, M, D) from the retain forward
            if token_risk_mask is not None:
                H = H[token_risk_mask]  # (T_risk, M, D)
            T = H.size(0)
            if T == 0:
                continue

            selected_M_f = selected_experts_per_layer[l_idx]
            frozen_M_r = [i for i in range(m.num_experts) if i not in selected_M_f]

            for expert_m in selected_M_f:
                for n in frozen_M_r:
                    Hm = H[:, expert_m, :]
                    Hn = H[:, n, :]

                    Hm_tilde = Hm / (torch.norm(Hm, p="fro") + 1e-8)
                    Hn_tilde = Hn / (torch.norm(Hn, p="fro") + 1e-8)

                    inner_product = torch.matmul(Hm_tilde.t(), Hn_tilde) / T
                    loss_sep = loss_sep + torch.norm(inner_product, p="fro") ** 2
        return loss_sep

    # ==========================================
    # core execution methods
    # ==========================================
    def learn(self, ckpt_path, ema_alpha=0.9):
        self.model._set_grad_mode("learning")
        use_ema = self.train_loader.batch_size <= 8
        ema_states = {}
        total_train_time = 0.0

        for epoch in range(self.num_epoch):
            self.model.train()
            epoch_start_time = time.time()

            running_total, running_ce, running_sp, running_bal, running_div = 0.0, 0.0, 0.0, 0.0, 0.0

            for batch in self.train_loader:
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                self.optimizer.zero_grad()
                logits, _ = self.model.forward_with_grad(images)

                # Fix (2): collect the DENSE routing distribution for sp/bal,
                # and the raw expert outputs for div.
                all_pi_dense, all_h = [], []
                moe_names = []
                for name, module in self.model.featurizer.model.named_modules():
                    if module.__class__.__name__ == "DeepMoELayer":
                        all_pi_dense.append(module.last_pi_all)   # (tokens, M)  <- dense
                        all_h.append(module.last_h)               # (tokens, M, D)
                        moe_names.append(name)

                ce_loss = self.criteria(logits, labels)
                sp_loss = sum(self._loss_sparse(pi) for pi in all_pi_dense) / len(all_pi_dense)
                bal_loss = sum(
                    self._loss_balance(pi, n, use_ema, ema_states, ema_alpha)
                    for pi, n in zip(all_pi_dense, moe_names)
                ) / len(all_pi_dense)
                div_loss = sum(self._loss_diversity(h) for h in all_h) / len(all_h)

                t_loss = (
                    ce_loss
                    + self.lambda_sparse * sp_loss
                    + self.lambda_balance * bal_loss
                    + self.lambda_div * div_loss
                )

                t_loss.backward()
                self.optimizer.step()

                running_total += t_loss.item()
                running_ce += ce_loss.item()
                running_sp += sp_loss.item()
                running_bal += bal_loss.item()
                running_div += div_loss.item()

            epoch_train_time = time.time() - epoch_start_time
            total_train_time += epoch_train_time

            num_batches = len(self.train_loader)
            avg_loss = running_total / num_batches

            print(f"[*] evaluating epoch {epoch+1}...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()

            print(
                f"epoch [{epoch+1}/{self.num_epoch}] | "
                f"total_loss: {avg_loss:.4f} (ce: {running_ce/num_batches:.4f}, "
                f"sp: {running_sp/num_batches:.4f}, bal: {running_bal/num_batches:.4f}, "
                f"div: {running_div/num_batches:.4f}) | "
                f"ra: {ra_score*100:.2f}% | fa: {fa_score*100:.2f}% | "
                f"ta: {ta_score*100:.2f}% | mia: {mia_score:.4f} | time: {epoch_train_time:.2f}s"
            )

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "ce_loss": running_ce / num_batches,
                    "retain_accuracy": ra_score,
                    "forget_accuracy": fa_score,
                    "test_accuracy": ta_score,
                    "mia_score": mia_score,
                }
            )

            torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_{epoch+1}.pt")

        peak_memory_gb = (
            torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )
        wandb.log({"total_train_time_sec": total_train_time, "peak_memory_gb": peak_memory_gb})

        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_train_time

    def unlearn(self, fa_threshold, ckpt_path):
        # frozen teacher for distillation (p_old).
        origin_model = copy.deepcopy(self.model)
        origin_model.eval()
        for param in origin_model.parameters():
            param.requires_grad = False

        moe_layers = [m for _, m in self.model.named_modules() if m.__class__.__name__ == "DeepMoELayer"]

        # ---- Fix (3): one-time forget-specific selection + optimizer scoped to M_f ----
        base_lr = self.optimizer.param_groups[0]["lr"]
        selected_experts_per_layer = self._select_forget_experts(moe_layers)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable_params, lr=base_lr)

        num_trainable = sum(p.numel() for p in trainable_params)
        print(f"[*] selected experts per layer (M_f): {selected_experts_per_layer}")
        print(f"[*] trainable params (chosen experts only): {num_trainable:,}")

        total_unlearn_time = 0.0

        for epoch in range(self.num_epoch):
            epoch_start_time = time.time()

            self.model.train()
            origin_model.eval()
            retain_iter = iter(self.retain_loader)

            total_loss_accum = 0.0
            n_risk_accum, n_seen_accum = 0, 0

            for forget_batch in self.forget_loader:
                images_f = forget_batch[0].to(self.device)
                labels_f = forget_batch[1].to(self.device)

                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(self.retain_loader)
                    retain_batch = next(retain_iter)

                images_r = retain_batch[0].to(self.device)
                labels_r = retain_batch[1].to(self.device)

                self.optimizer.zero_grad()

                # --- forgetting term (D_f) ---
                logits_f, _ = self.model.forward_with_grad(images_f)
                loss_forget = self._unlearn_loss_forget(logits_f, labels_f)

                # --- retain forward (populates caches for the at-risk computation) ---
                logits_r, _ = self.model.forward_with_grad(images_r)

                # Fix (1): restrict retain / distill / separation to D_r^risk.
                risk_mask = self._compute_risk_mask(moe_layers, selected_experts_per_layer)  # (B_r,)
                n_seen_accum += risk_mask.numel()
                n_risk_accum += int(risk_mask.sum().item())

                if risk_mask.any():
                    S = moe_layers[0].last_topk_indices.size(1)
                    token_risk_mask = risk_mask.repeat_interleave(S)  # (B_r * S,)

                    logits_r_risk = logits_r[risk_mask]
                    labels_r_risk = labels_r[risk_mask]
                    images_r_risk = images_r[risk_mask]

                    loss_retain = self._unlearn_loss_retain(logits_r_risk, labels_r_risk)
                    loss_distill = self._unlearn_loss_distill(logits_r_risk, images_r_risk, origin_model)
                    loss_sep = self._unlearn_loss_separation(
                        moe_layers, selected_experts_per_layer, token_risk_mask
                    )
                else:
                    zero = torch.zeros((), device=self.device)
                    loss_retain = loss_distill = loss_sep = zero

                total_loss = (
                    loss_forget
                    + self.beta * loss_retain
                    + self.gamma * loss_distill
                    + self.eta * loss_sep
                )
                total_loss.backward()
                self.optimizer.step()

                total_loss_accum += total_loss.item()

            avg_loss = total_loss_accum / len(self.forget_loader)
            risk_frac = n_risk_accum / max(n_seen_accum, 1)
            epoch_time = time.time() - epoch_start_time
            total_unlearn_time += epoch_time

            print(f"[*] evaluating epoch {epoch+1}...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()

            print(f"--> Epoch [{epoch+1}/{self.num_epoch}] | Time: {epoch_time:.2f}s | Loss: {avg_loss:.4f}")
            print(
                f"--> Metrics: RA: {ra_score*100:.2f}% | FA: {fa_score*100:.2f}% | "
                f"TA: {ta_score*100:.2f}% | MIA: {mia_score:.4f} | at-risk frac: {risk_frac:.3f}"
            )
            print("-" * 40)

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "unlearn_loss": avg_loss,
                    "ra": ra_score,
                    "fa": fa_score,
                    "ta": ta_score,
                    "mia": mia_score,
                    "at_risk_fraction": risk_frac,
                }
            )

            torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_{epoch+1}.pt")

            if fa_score <= fa_threshold:
                print(f"[*] early stopping triggered at epoch {epoch+1} (FA <= {fa_threshold})")
                break

        peak_memory_gb = (
            torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )
        wandb.log({"total_unlearn_time_sec": total_unlearn_time, "peak_memory_gb": peak_memory_gb})

        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_unlearn_time