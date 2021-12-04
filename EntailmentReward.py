from transformers import Seq2SeqTrainer, AutoTokenizer, AlbertForSequenceClassification

import torch

from RE2.src.evaluator import Evaluator

class EntailmentReward(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fnct = torch.nn.CrossEntropyLoss()
        self.tokenizer_entail = AutoTokenizer.from_pretrained('textattack/albert-base-v2-snli')
        self.model_entail = AlbertForSequenceClassification.from_pretrained('textattack/albert-base-v2-snli')
        self.device = torch.device("cuda")
        for param in self.model_entail.parameters():
          param.requires_grad = False
        self.model_entail.to(self.device)

        # Evaluator(model_path="textattack/albert-base-v2-snli", data_file=None)

    def compute_loss(self, model, inputs, return_outputs=False):
        # implement entailment scoring here
        # non-differentiable because of model generation? how to handle this?
        # might just work naively?
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        gen_outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=56,
                                     output_scores=True, return_dict_in_generate=True)

        self_loss = torch.mean(gen_outputs['sequences_scores'])
        gen_outputs = gen_outputs['sequences']
        with torch.no_grad():
            gen_outputs = self.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
            input_docs = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

            entail_pairs = []
            pair_len = [0]
            for i in range(len(gen_outputs)):
                curr_pairs = [[gen_outputs[i], sent] for sent in input_docs[i].split("\n")]
                entail_pairs.extend(curr_pairs)
                pair_len.append(pair_len[-1] + len(curr_pairs))
            reward = self.inference_score(entail_pairs)
            reward = torch.FloatTensor([1-torch.max(reward[pair_len[i-1]:pair_len[i]]) for i in range(1, len(pair_len))])

            # to try: reward using different calculation, normalized reward
            reward = torch.max(reward)

            reward = reward.to(self.device)

        loss += reward * self_loss

        return (loss, outputs) if return_outputs else loss


    def inference_score(self, sent_pairs):
        softmax = torch.nn.Softmax(dim=1)
        inputs = self.tokenizer_entail(sent_pairs, return_tensors="pt", padding=True).to(self.device)
        #inputs = [self.tokenizer_entail(datum, return_tensors="pt", padding=True) for datum in sent_pairs]
        outputs = self.model_entail(**inputs)
        logits = outputs.logits
        #return logits[:, 2]
        return 1 - softmax(logits)[:, 0] # contradiction score


