import torch
from .MyTokenizer  import Tokenizer
from nets.MyTransformer import TransformerEncoder,TransfomerDecoder,MYTransFormer
from .MyTokenizer import get_tokenizer
# encoder = TransformerEncoder(2,2,vocab_size=54000)
# decoder = TransfomerDecoder(2,2,vocab_size=54000)
# tokenizer =  get_tokenizer(True,language="en", task="translate")

def infer_encoder_decoder_once(Transformer:MYTransFormer, tokenizer:Tokenizer,encoder_input:str,device="cuda"):

        y_hat = Transformer(torch.tensor([tokenizer.encode(encoder_input)]).to(device),torch.tensor([[tokenizer.sot]+tokenizer.encode("我草拟吗！傻逼东西！")]).to(device))

        return tokenizer.decode(torch.argmax(y_hat[0],dim=2)[0].to("cpu"))

def infer_encoder_decoder_greddy(Transformer:MYTransFormer, tokenizer:Tokenizer,encoder_input:str,max_lens=256,device="cuda"):


        decode_list = []
        decode_list.append(tokenizer.sot)
        input_x = torch.tensor([decode_list]).to(device)

        y_hat = Transformer(torch.tensor([tokenizer.encode(encoder_input)]).to(device),input_x)

        last= torch.argmax(y_hat[0], dim=2)[0][-1].to("cpu").item()

        counter= 0

        while last!= tokenizer.eot:

                decode_list.append(last)

                input_x = torch.tensor([decode_list]).to(device)

                y_hat =Transformer(torch.tensor([tokenizer.encode(encoder_input)]).to(device),input_x)

                # print(torch.argmax(y_hat[0], dim=2).shape)

                last = torch.argmax(y_hat[0], dim=2)[:,-1].to("cpu").item()

                counter+=1

                if counter>max_lens:

                        break
        return tokenizer.decode(decode_list)






#
# x="hello"
#
# y="你好"
#
# print(infer_encoder_decoder_greddy(encoder,decoder,tokenizer,x,device="cpu"))

