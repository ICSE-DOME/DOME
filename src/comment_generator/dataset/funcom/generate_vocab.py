from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.decoders import BPEDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit

file_list = ['./train/ast.train', './valid/ast.valid', './test/ast.test',
             './train/comment.train', './valid/comment.valid', './test/comment.test']

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.decoder = BPEDecoder()
tokenizer.pre_tokenizer = WhitespaceSplit()

trainer = BpeTrainer(vocab_size=50000, special_tokens=["[PAD]", "[UNK]", "[WHAT/]", "[/WHAT]", "[WHY/]", "[/WHY]",
                                                       "[USAGE/]", "[/USAGE]", "[DONE/]", "[/DONE]", "[PROP/]",
                                                       "[/PROP]", "[EOS]"],
                     end_of_word_suffix='</w>', show_progress=True)

tokenizer.train(file_list, trainer)
tokenizer.save('./bpe_tokenizer.json')

tokenizer = Tokenizer.from_file('./bpe_tokenizer.json')
print(tokenizer.encode("automatedtest ( )").tokens)
