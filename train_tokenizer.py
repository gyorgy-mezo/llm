from datasets import load_dataset
from transformers import AutoTokenizer

example = 'Calvertron (Alex Calver) a kétezres évek kezdete óta készít zenéket, ' \
          'elképesztő tempóban: több mint 100 megjelenést tudhat magáénak eddig, ' \
          'és ez a szám hétről-hétre nő.\n' \
          'Calver döngetős technóval kezdte, ' \
          'majd mindenféle kitérők után jutott el a mostani electro-/fidget-/bassline-house hibridhangzásáig. ' \
          'Tagja volt a rövidéletű, ámde rendkívül sikeres Twocker duónak, ' \
          'amely 2008-ban sorozatban gyártotta a mocskos, ' \
          'röfögős house-fajzatokat - ' \
          'nem hivatalos remixük CJ Bolland Sugar Is Sweeteréből az év egyik legnagyobb klubslágere volt. ' \
          'Amióta önállósította magát, ' \
          'a saját zenék frontján letudott egy-egy megjelenést ' \
          'a két fő "post-fidget" house-kiadónál (Wearhouse, Potty Mouth), ' \
          'remixelte az indie-dance csillag Friendly Fires-t, ' \
          'és önálló kiadót alapított (Jack Knife Records). ' \
          'Folyamatos naprakészségét jelzi, ' \
          'hogy hamarosan napvilágot lát Rico Tubbs-szal közös EP-je ' \
          'az amerikai Trouble & Bass árulkodó című Heavy Bass Champions Of The World sorozatában, ' \
          'illetve egy ghetto-house-os darab Tim Healey-vel közösen, ' \
          'annak Giant Pussy labelén.\n' \
          'Az OTHERSIDE egy havonta jelentkező partysorozat a BASSZUS és a TÖRTRITMUS szerelmeseinek, ' \
          'amely elsőként hozza el a londoni underground klubestek legmenőbb műfajait a budapesti éjszakákba. ' \
          'Olyan stílusok külföldi úttörői kapnak meghívást a bulikra, ' \
          'mint a szinte már a modern brit zene jelképévé váló DUBSTEP, ' \
          'az ebből és a DNB-ből építkező DRUMSTEP, a reszelős FILTHSTEP, ' \
          'az angol hip-hopból gyökeredző GRIME, valamint az ELECTRO és a FIDGET.'


def test_tokenizer(tokenizer):
    tokens = tokenizer.tokenize(example)
    print(len(tokens))
    print(tokens)


def tested_pretrained_tokenizer(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name)
    test_tokenizer(tokenizer)
    return tokenizer


# tested_pretrained_tokenizer("gpt2")
# tested_pretrained_tokenizer("hun_tokenizer_1")
# tested_pretrained_tokenizer("hun_tokenizer_32k")

do_train = True

if do_train:
    # raw_datasets = load_dataset("code_search_net", "python")
    # raw_datasets = load_dataset("oscar", "unshuffled_deduplicated_hu")
    raw_datasets = load_dataset("mc4", "hu")
    print(raw_datasets["train"])

    print(raw_datasets["train"][0])
    print(raw_datasets["train"][6582908 - 1])


    def get_training_corpus():
        dataset = raw_datasets["train"]
        for start_idx in range(0, len(dataset), 1000):
            samples = dataset[start_idx: start_idx + 1000]
            yield samples["text"]


    training_corpus = get_training_corpus()

    original_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    trained_tokenizer = original_tokenizer.train_new_from_iterator(training_corpus, 32000)

    trained_tokenizer.save_pretrained("hun_tokenizer_mc4_32k")

    test_tokenizer(trained_tokenizer)
