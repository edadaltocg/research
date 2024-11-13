import pysnooper
from transformers import AutoTokenizer

from dnn.data import preprocess_text
from dnn.tokenizer import Tokenizer, train_sp_tokenizer_from_iterator


@pysnooper.snoop()
def test_sp_tokenizer():
    filename = "examples/botchan.txt"
    with open(filename) as f:
        text = f.read()
    # chunk text
    chunk_size = 1000
    chunks = [
        " ".join(preprocess_text(text[i : i + chunk_size]))
        for i in range(0, len(text), chunk_size)
    ]
    iterator = iter(chunks)

    # train
    tokenizer = train_sp_tokenizer_from_iterator(
        iterator, prefix="test_sp", vocab_size=150
    )
    tokenizer = Tokenizer("output/tokenizers/test_sp.model")

    # text encode/decode
    text = "ipsum dolor sit amet."
    encoded_text = tokenizer.encode(text)
    decoded_text = tokenizer.decode(encoded_text)
    assert decoded_text == text

    # list of texts encode/decode
    list_of_texts = [
        "ipsum dolor sit amet.",
        "qui minim labore adipisicing minim sint cillum sint consectetur cupidatat.",
    ]
    encoded_texts = tokenizer.encode(list_of_texts)
    decoded_texts = tokenizer.decode(encoded_texts)
    print(decoded_texts)
    assert decoded_texts == list_of_texts

    # check paddings
    encoded_texts = tokenizer.encode(list_of_texts, max_len=10)
    assert encoded_texts.shape == (2, 10)
    print(encoded_texts)

    attn_mask = tokenizer.get_attn_mask(encoded_texts)
    print(attn_mask)

    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    hf_tokenizer.pad_token = "[PAD]"
    encoded = hf_tokenizer(list_of_texts, return_tensors="pt", padding=True)
    print(encoded)


@pysnooper.snoop()
def test_text_processing():
    text = "Lorem ipsum dolor sit amet, qui minim labore adipisicing minim sint cillum sin.1234567890"
    preprocessed_text = preprocess_text(text)
    print(preprocessed_text)


def test_tokenize_code():
    code = """
def distrib_spawn(fn, *args):
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fn, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
    print(encoded)

    tokenizer = Tokenizer("output/tokenizers/test_sp.model")
    encoded = tokenizer.raw_encode(code, out_type=str)
    print(encoded)
