import torch

import torch.nn.functional as F





_TOKEN_START_INDEX = 0





def _validate_2d_tensor(name, tensor):

    if tensor.dim() != 2:

        raise ValueError(f"{name} must be a 2D tensor, got shape={tuple(tensor.shape)}")





def _validate_3d_tensor(name, tensor):

    if tensor.dim() != 3:

        raise ValueError(f"{name} must be a 3D tensor, got shape={tuple(tensor.shape)}")





def build_caption_mask(token_ids):

                                                            



                      

                                       

                                            

                                 

                                       



         

                                         



            

                                                                            

                                

       

    _validate_2d_tensor("token_ids", token_ids)



    device = token_ids.device

    batch_size, seq_len = token_ids.shape

    mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)



                                                                                    

                                               

    eot_idx = token_ids.argmax(dim=-1)



    for row in range(batch_size):

        end = int(eot_idx[row].item())

        start = _TOKEN_START_INDEX + 1

        if end > start:

            mask[row, start:end] = 1.0



    return mask





def cosine_cost_matrix(x, y, eps=1e-12):

                                             



         

                                    

                                    

                                                          



            

                                                                           

       

    _validate_3d_tensor("x", x)

    _validate_3d_tensor("y", y)



    if x.shape[0] != y.shape[0]:

        raise ValueError(

            f"x and y must share the same batch size, got {x.shape[0]} and {y.shape[0]}"

        )

    if x.shape[2] != y.shape[2]:

        raise ValueError(

            f"x and y must share the same feature dimension, got {x.shape[2]} and {y.shape[2]}"

        )



    x_norm = F.normalize(x, dim=-1, eps=eps)

    y_norm = F.normalize(y, dim=-1, eps=eps)

    sim = torch.bmm(x_norm, y_norm.transpose(1, 2))

    return 1.0 - sim





def build_uniform_target_mass(batch_size, num_targets, device, dtype):

                                                                      

    if num_targets <= 0:

        raise ValueError(f"num_targets must be positive, got {num_targets}")

    return torch.full((batch_size, num_targets), 1.0 / num_targets, device=device, dtype=dtype)





def build_source_mass(src_mask):

                                                                      

    _validate_2d_tensor("src_mask", src_mask)



    src_mask = src_mask.to(dtype=torch.float32)

    mass_sum = src_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

    return src_mask / mass_sum





def build_similarity_source_mass(sim, src_mask, temperature=0.1, detach=True):

                                                                                      

    _validate_3d_tensor("sim", sim)

    _validate_2d_tensor("src_mask", src_mask)



    if sim.shape[:2] != src_mask.shape:

        raise ValueError(

            f"sim and src_mask shapes are incompatible, got sim={tuple(sim.shape)} "

            f"src_mask={tuple(src_mask.shape)}"

        )

    if temperature <= 0:

        raise ValueError(f"temperature must be positive, got {temperature}")



    if detach:

        sim = sim.detach()



    src_mask = src_mask.to(device=sim.device, dtype=torch.float32)

    token_score = sim.max(dim=-1).values.to(dtype=torch.float32)    

    token_score = token_score.masked_fill(src_mask <= 0, -1e4)



    src_mass = F.softmax(token_score / temperature, dim=-1)

    src_mass = src_mass * src_mask

    mass_sum = src_mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    return src_mass / mass_sum





def sinkhorn_logsumexp(cost, a, b, eps=0.1, max_iter=20):

                                                                               



         

                                       

                                          

                                          

                                                

                                               



            

                                            

       

    _validate_3d_tensor("cost", cost)

    if a.dim() != 2:

        raise ValueError(f"a must be a 2D tensor, got shape={tuple(a.shape)}")

    if b.dim() != 2:

        raise ValueError(f"b must be a 2D tensor, got shape={tuple(b.shape)}")

    if cost.shape[:2] != a.shape:

        raise ValueError(

            f"cost and a shapes are incompatible, got cost={tuple(cost.shape)} a={tuple(a.shape)}"

        )

    if cost.shape[0] != b.shape[0] or cost.shape[2] != b.shape[1]:

        raise ValueError(

            f"cost and b shapes are incompatible, got cost={tuple(cost.shape)} b={tuple(b.shape)}"

        )

    if eps <= 0:

        raise ValueError(f"eps must be positive, got {eps}")

    if max_iter <= 0:

        raise ValueError(f"max_iter must be positive, got {max_iter}")



    cost = cost.to(dtype=torch.float32)

    a = a.to(device=cost.device, dtype=torch.float32).clamp_min(1e-8)

    b = b.to(device=cost.device, dtype=torch.float32).clamp_min(1e-8)



    log_a = torch.log(a)

    log_b = torch.log(b)

    log_k = -cost / eps



    u = torch.zeros_like(log_a)

    v = torch.zeros_like(log_b)



    for _ in range(max_iter):

        u = log_a - torch.logsumexp(log_k + v.unsqueeze(1), dim=2)

        v = log_b - torch.logsumexp(log_k + u.unsqueeze(2), dim=1)



    return torch.exp(log_k + u.unsqueeze(2) + v.unsqueeze(1))





def _resolve_source_mass(cost, src_mask=None, src_mass=None):

    if src_mass is None and src_mask is None:

        raise ValueError("Either src_mask or src_mass must be provided")



    if src_mass is not None:

        _validate_2d_tensor("src_mass", src_mass)

        if cost.shape[:2] != src_mass.shape:

            raise ValueError(

                f"cost and src_mass shapes are incompatible, got cost={tuple(cost.shape)} "

                f"src_mass={tuple(src_mass.shape)}"

            )

        return src_mass.to(device=cost.device, dtype=cost.dtype)



    _validate_2d_tensor("src_mask", src_mask)

    if cost.shape[:2] != src_mask.shape:

        raise ValueError(

            f"cost and src_mask shapes are incompatible, got cost={tuple(cost.shape)} "

            f"src_mask={tuple(src_mask.shape)}"

        )

    return build_source_mass(src_mask).to(device=cost.device, dtype=cost.dtype)





def masked_ot_distance(cost, src_mask=None, src_mass=None, eps=0.1, max_iter=20):

                                                                    



         

                                       

                                                                                          

                                                                                      

                                                

                                               



            

                                     

                                            

       

    _validate_3d_tensor("cost", cost)



    cost = cost.to(dtype=torch.float32)

    src_mass = _resolve_source_mass(cost=cost, src_mask=src_mask, src_mass=src_mass)

    tgt_mass = build_uniform_target_mass(

        batch_size=cost.shape[0],

        num_targets=cost.shape[2],

        device=cost.device,

        dtype=cost.dtype,

    )



    transport = sinkhorn_logsumexp(cost, src_mass, tgt_mass, eps=eps, max_iter=max_iter)

    distance = (transport * cost).sum(dim=(1, 2))

    return distance, transport





def compute_ot_transport(cost, src_mask=None, src_mass=None, eps=0.1, max_iter=20):

                                                                   

    _validate_3d_tensor("cost", cost)



    cost = cost.to(dtype=torch.float32)

    src_mass = _resolve_source_mass(cost=cost, src_mask=src_mask, src_mass=src_mass)

    tgt_mass = build_uniform_target_mass(

        batch_size=cost.shape[0],

        num_targets=cost.shape[2],

        device=cost.device,

        dtype=cost.dtype,

    )

    transport = sinkhorn_logsumexp(cost, src_mass, tgt_mass, eps=eps, max_iter=max_iter)

    return transport, tgt_mass





def transport_barycentric_projection(src_seq, transport, tgt_mass, src_mask=None):

                                                                              

    _validate_3d_tensor("src_seq", src_seq)

    _validate_3d_tensor("transport", transport)

    _validate_2d_tensor("tgt_mass", tgt_mass)

    if src_seq.shape[:2] != transport.shape[:2]:

        raise ValueError(

            f"src_seq and transport shapes are incompatible, got "

            f"src_seq={tuple(src_seq.shape)} transport={tuple(transport.shape)}"

        )

    if transport.shape[0] != tgt_mass.shape[0] or transport.shape[2] != tgt_mass.shape[1]:

        raise ValueError(

            f"transport and tgt_mass shapes are incompatible, got "

            f"transport={tuple(transport.shape)} tgt_mass={tuple(tgt_mass.shape)}"

        )



    src_seq = src_seq.to(dtype=torch.float32)

    transport = transport.to(device=src_seq.device, dtype=src_seq.dtype)

    tgt_mass = tgt_mass.to(device=src_seq.device, dtype=src_seq.dtype)



    if src_mask is not None:

        _validate_2d_tensor("src_mask", src_mask)

        if src_mask.shape != src_seq.shape[:2]:

            raise ValueError(

                f"src_mask and src_seq shapes are incompatible, got "

                f"src_mask={tuple(src_mask.shape)} src_seq={tuple(src_seq.shape)}"

            )

        src_seq = src_seq * src_mask.to(device=src_seq.device, dtype=src_seq.dtype).unsqueeze(-1)



    aligned = torch.bmm(transport.transpose(1, 2), src_seq)

    return aligned / tgt_mass.unsqueeze(-1).clamp_min(1e-8)

