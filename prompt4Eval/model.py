from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch.nn as nn
import torch

class LMPrompt4Eval(nn.Module):
    def __init__(self, model_name, answer_ids, args):
        super(LMPrompt4Eval, self).__init__()
        self.model_name = model_name
        self.LM = AutoModelForMaskedLM.from_pretrained(model_name)
        #self.LM.resize_token_embeddings(args.vocab_size)
        self.tok = AutoTokenizer.from_pretrained(model_name)

        for param in self.LM.parameters():
            param.requires_grad = True

        self.answer_ids = answer_ids
        self.mask_token_id = self.tok.mask_token_id
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, batch_enc, batch_attn, batch_labs):
        outputs = self.LM(input_ids=batch_enc,
                            attention_mask=batch_attn)
        out_logits = outputs.logits

        mask_position = batch_enc.eq(self.mask_token_id)

        mask_logits = out_logits[mask_position, :].view(out_logits.size(0), -1, out_logits.size(-1))[:, -1, :]

        answer_logits = mask_logits[:, self.answer_ids]

        batch_labs = batch_labs.view(-1).long()

        loss = self.loss_func(answer_logits, batch_labs)

        return loss, answer_logits.softmax(dim=1)
