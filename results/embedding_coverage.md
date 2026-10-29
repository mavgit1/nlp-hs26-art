# Embedding coverage on train

| embedding | tokenizer | type coverage | token coverage | token coverage + lowercase fallback |
|---|---|---|---|---|
| fasttext | cased/split | 98.2% | 99.4% | 99.5% |
| fasttext | cased/nosplit | 92.7% | 99.0% | 99.0% |
| fasttext | lower/split | 97.4% | 99.4% | 99.4% |
| fasttext | lower/nosplit | 91.8% | 98.9% | 98.9% |
| glove | cased/split | 98.1% | 99.9% | 99.9% |
| glove | cased/nosplit | 92.6% | 99.0% | 99.0% |
| glove | lower/split | 97.6% | 99.9% | 99.9% |
| glove | lower/nosplit | 91.9% | 98.9% | 98.9% |
| word2vec | cased/split | 94.3% | 78.3% | 78.3% |
| word2vec | cased/nosplit | 89.5% | 78.1% | 78.2% |
| word2vec | lower/split | 89.1% | 77.6% | 77.6% |
| word2vec | lower/nosplit | 84.5% | 77.4% | 77.4% |

## Most frequent missing tokens

- **fasttext cased/split**: `n't` (3225), `Shalil` (4), `Tanice` (4), `BunnBunn` (4), `Loraline` (4), `Benisha` (4), `saded` (3), `forgeted` (3), `inessentialed` (3), `Trichelle` (3), `Eerin` (3), `Tuie` (3), `SnowGirl` (3), `Lamron` (3), `Kenden` (3)
- **fasttext cased/nosplit**: `couldn't` (605), `wasn't` (437), `Gina's` (192), `wouldn't` (130), `hadn't` (88), `Year's` (88), `Amy's` (85), `friend's` (85), `Bob's` (68), `he'd` (59), `Tim's` (56), `Tom's` (55), `Mary's` (50), `Jim's` (49), `she'd` (47)
- **fasttext lower/split**: `n't` (3225), `lewy` (7), `zaiden` (7), `hailie` (6), `antonela` (5), `fariah` (5), `jorie` (5), `myah` (4), `mahlon` (4), `seaver` (4), `shalil` (4), `stassi` (4), `jacelyn` (4), `klea` (4), `lamay` (4)
- **fasttext lower/nosplit**: `couldn't` (605), `wasn't` (437), `gina's` (194), `wouldn't` (130), `year's` (98), `hadn't` (88), `amy's` (85), `friend's` (85), `bob's` (68), `i'm` (65), `he'd` (61), `tim's` (56), `tom's` (55), `she'd` (52), `mary's` (50)
- **glove cased/split**: `�` (8), `Fariah` (5), `Shalil` (4), `'t` (4), `Marisse` (4), `Tanice` (4), `BunnBunn` (4), `'mores` (4), `Loraline` (4), `Benisha` (4), `forgeted` (3), `'clock` (3), `inessentialed` (3), `Trichelle` (3), `Eerin` (3)
- **glove cased/nosplit**: `couldn't` (605), `wasn't` (437), `Gina's` (192), `wouldn't` (130), `hadn't` (88), `Year's` (88), `Amy's` (85), `friend's` (85), `Bob's` (68), `he'd` (59), `Tim's` (56), `Tom's` (55), `Mary's` (50), `Jim's` (49), `she'd` (47)
- **glove lower/split**: `�` (8), `zaiden` (7), `antonela` (5), `fariah` (5), `mahlon` (4), `shalil` (4), `'t` (4), `kelsea` (4), `stassi` (4), `jacelyn` (4), `klea` (4), `lamay` (4), `marisse` (4), `quex` (4), `drita` (4)
- **glove lower/nosplit**: `couldn't` (605), `wasn't` (437), `gina's` (194), `wouldn't` (130), `year's` (98), `hadn't` (88), `amy's` (85), `friend's` (85), `bob's` (68), `i'm` (65), `he'd` (61), `tim's` (56), `tom's` (55), `she'd` (52), `mary's` (50)
- **word2vec cased/split**: `.` (65575), `to` (23004), `a` (19733), `and` (13074), `,` (6077), `of` (5694), `'s` (4818), `!` (1777), `"` (139), `'` (81), `10` (69), `20` (64), `didnt` (53), `100` (44), `-` (43)
- **word2vec cased/nosplit**: `.` (65575), `to` (23004), `a` (19733), `and` (13074), `,` (6077), `of` (5694), `!` (1777), `Gina's` (192), `"` (139), `Year's` (88), `Amy's` (85), `friend's` (85), `'` (82), `10` (69), `Bob's` (68)
- **word2vec lower/split**: `.` (65575), `to` (23081), `a` (20886), `and` (13197), `,` (6077), `of` (5694), `'s` (4821), `!` (1777), `"` (139), `'` (81), `10` (69), `20` (64), `didnt` (54), `tami` (53), `maddie` (46)
- **word2vec lower/nosplit**: `.` (65575), `to` (23081), `a` (20883), `and` (13197), `,` (6077), `of` (5694), `!` (1777), `gina's` (194), `"` (139), `amy's` (85), `friend's` (85), `'` (82), `10` (69), `bob's` (68), `i'm` (65)
