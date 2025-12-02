import torch
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

def evaluate(model, dataloader, tag_list, device):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    outputs = []

    with torch.no_grad():
        for batch in dataloader:
            x, y, mask = batch
            x, y = x.to(device), y.to(device)
            mask = mask.to(device)

            logits = model(x, mask)
            preds = logits.argmax(dim=-1)

            # iterate each sentence
            for i in range(len(preds)):
                seq_len = int(mask[i].sum().item())

                # predicted tags
                pred_seq = preds[i][:seq_len].tolist()
                pred_tags = [tag_list[idx] for idx in pred_seq]

                # true tags
                true_seq = y[i][:seq_len].tolist()
                true_tags = [tag_list[idx] for idx in true_seq]

                all_preds.append(pred_tags)
                all_labels.append(true_tags)
                tags_str = " ".join(pred_tags)
                outputs.append([i+1, tags_str])

    f1 = f1_score(all_labels, all_preds, mode="strict", scheme=IOB2)
    return f1, outputs