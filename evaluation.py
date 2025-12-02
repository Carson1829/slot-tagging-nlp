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
        for x, y, mask in dataloader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1).cpu()

            # iterate each sentence
            for i in range(len(preds)):
                seq_len = int(mask[i].sum().item())

                # predicted tags
                pred_seq = preds[i][:seq_len].tolist()
                pred_tags = [tag_list[idx] for idx in pred_seq]

                # true tags
                gold_seq = y[i][:seq_len].tolist()
                gold_tags = [tag_list[idx] for idx in gold_seq]

                all_preds.append(pred_tags)
                all_labels.append(gold_tags)
                tags_str = " ".join(pred_tags)
                outputs.append([i+1, tags_str])

    f1 = f1_score(all_labels, all_preds, mode="strict", scheme=IOB2)
    return f1, outputs