def calculate_penalty(curr_action, prev_action, delta_max):
    if prev_action == -1:
        return 0.0

    vp_prev = prev_action // 5
    iv_prev = prev_action % 5
    vp_curr = curr_action // 5
    iv_curr = curr_action % 5

    # Calculate absolute change in dosage levels
    delta_vp = abs(vp_curr - vp_prev)
    delta_iv = abs(iv_curr - iv_prev)
    penalty = 0.0

    if delta_vp > delta_max or delta_iv > delta_max:
        penalty = 1.0

    return penalty
