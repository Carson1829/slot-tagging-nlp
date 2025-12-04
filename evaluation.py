import torch
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2


def get_preds(model, dataloader, tag_list, device):
    model.to(device)
    model.eval()

    outputs = []     # final list of [id, predicted tag string]
    id_ = 1           # running sentence ID counter

    with torch.no_grad():
        for batch in dataloader:
            x, y, mask = batch
            x, y = x.to(device), y.to(device)
            mask = mask.to(device)

            # Forward pass: logits → (B, L, num_tags)
            logits = model(x, mask)

            # Take argmax over tag dimension to get predicted tag IDs
            preds = logits.argmax(dim=-1)   # (B, L)

            for i in range(len(preds)):
                # Compute the actual length for prediction slicing
                seq_len = int(mask[i].sum().item())
                pred_seq = preds[i][:seq_len].tolist()

                # Convert tag IDs back to tag strings
                pred_tags = [tag_list[idx] for idx in pred_seq]

                tags_str = " ".join(pred_tags)
                outputs.append([id_, tags_str])

                id_ += 1

    return outputs  # list of [id, "TAG1 TAG2 TAG3 ..."]