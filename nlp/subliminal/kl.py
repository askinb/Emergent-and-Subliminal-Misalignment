"""KL losses for the distillation methods.

- forward_kl_loss: OPTD. Full-vocab KL(p_T || p_S) over response-token positions
  on a teacher-forced trajectory. Mode-covering. Off-policy.
- token_rev_kl_loss: OPD. Per-token Monte Carlo reverse-KL surrogate over a
  student-sampled trajectory. Mode-seeking. On-policy.
- sample_rollouts: student.generate() helper for OPD.
"""

import torch
import torch.nn.functional as F

from nlp.subliminal.common import add_special_tokens, render_chat_prompt


@torch.no_grad()
def sample_rollouts(student, tok, prompts, max_new: int,
                    temperature: float, top_p: float, max_seq: int):
    """Generate new student completions for `prompts`. Returns
    (input_ids, attention_mask, prompt_len) where prompt_len is the
    (left-padded) prompt length P."""
    student.eval()
    tok.padding_side = "left"
    formatted = [render_chat_prompt(tok, [{"role": "user", "content": p}]) for p in prompts]
    enc = tok(formatted, return_tensors="pt", padding=True, truncation=True,
              max_length=max_seq - max_new,
              add_special_tokens=add_special_tokens(tok, formatted))
    enc = {k: v.to(student.device) for k, v in enc.items()}
    P = enc["input_ids"].shape[1]
    out = student.generate(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
        max_new_tokens=max_new, do_sample=True,
        temperature=temperature, top_p=top_p,
        pad_token_id=tok.pad_token_id, use_cache=True,
    )
    attn = torch.ones_like(out)
    attn[:, :P] = enc["attention_mask"]
    attn[:, P:] = (out[:, P:] != tok.pad_token_id).long()
    return out, attn, P


def forward_kl_loss(teacher, student, ids, attn, resp_mask):
    """Full-vocab KL(p_T || p_S) on a teacher-forced trajectory.

    Loss = mean_t [ sum_v p_T(v|x_<t) * (log p_T(v|x_<t) - log p_S(v|x_<t)) ].
    p_T detached, p_S live. resp_mask selects the response-token positions.
    """
    teacher.eval()
    with torch.no_grad():
        t_logits = teacher(input_ids=ids, attention_mask=attn).logits
    student.train()
    s_logits = student(input_ids=ids, attention_mask=attn).logits

    log_ps = F.log_softmax(s_logits[:, :-1, :].float(), dim=-1)
    log_pt = F.log_softmax(t_logits[:, :-1, :].float(), dim=-1)
    p_t = log_pt.exp().detach()
    kl_pos = (p_t * (log_pt.detach() - log_ps)).sum(dim=-1)
    denom = resp_mask.sum().clamp(min=1.0)
    loss = (kl_pos * resp_mask).sum() / denom

    with torch.no_grad():
        rev = ((log_ps.exp() * (log_ps - log_pt)).sum(dim=-1) * resp_mask).sum() / denom
        fwd = ((log_pt.exp() * (log_pt - log_ps)).sum(dim=-1) * resp_mask).sum() / denom
    return loss, {"rev_kl": rev.item(), "fwd_kl": fwd.item(),
                  "resp_tok": int(denom.item())}


def token_rev_kl_loss(teacher, student, ids, attn, resp_mask):
    """Per-token Monte Carlo reverse-KL surrogate on a student-sampled trajectory.

    REINFORCE-style:
        rho_t = sg(log p_S(x_t)) - log p_T(x_t)
        L     = mean_t [ rho_t * log p_S(x_t) ]   (only log p_S has grad)
    Gradient = mean_t [ rho_t * grad log p_S(x_t) ], a single-sample Monte Carlo
    estimate of the gradient of E_{x~p_S} KL(p_S || p_T) per token.
    """
    teacher.eval()
    with torch.no_grad():
        t_logits = teacher(input_ids=ids, attention_mask=attn).logits[:, :-1, :]
        log_pt = F.log_softmax(t_logits.float(), dim=-1)
    student.train()
    s_logits = student(input_ids=ids, attention_mask=attn).logits[:, :-1, :]
    log_ps = F.log_softmax(s_logits.float(), dim=-1)

    next_ids = ids[:, 1:].unsqueeze(-1)
    log_ps_x = log_ps.gather(-1, next_ids).squeeze(-1)
    with torch.no_grad():
        log_pt_x = log_pt.gather(-1, next_ids).squeeze(-1)

    rho = log_ps_x.detach() - log_pt_x
    denom = resp_mask.sum().clamp(min=1.0)
    loss = (rho * log_ps_x * resp_mask).sum() / denom

    with torch.no_grad():
        mc_kl = (rho * resp_mask).sum() / denom
        rho_abs = (rho.abs() * resp_mask).sum() / denom
    return loss, {"mc_rev_kl": mc_kl.item(), "rho_abs": rho_abs.item(),
                  "resp_tok": int(denom.item())}
