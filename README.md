# Bielik LoRA Lab

Lokalne uruchamianie Bielika przez Ollama, przygotowanie korpusow w PostgreSQL z pgvector oraz trening LoRA/QLoRA dla `speakleash/Bielik-11B-v3.0-Instruct`.

## Docker Compose

Podstawowy stack uruchamia PostgreSQL 16 z rozszerzeniem `pgvector`, migracje Liquibase XML, API oraz GUI. Nie uruchamia treningu ani nie obciaza GPU.

```powershell
docker compose up --build
```

GUI jest dostepne pod `http://localhost:5173`, API pod `http://localhost:8000/docs`, a PostgreSQL na porcie `5432`. Korpusy sa przechowywane w PostgreSQL i eksportowane z GUI jako JSONL dla wybranego podzialu.

## Frontend lokalnie

Przy dzialajacym API na porcie `8000` uruchom GUI z hot reloadem:

```powershell
Set-Location frontend
npm install
npm run dev
```

Vite udostepnia GUI na `http://localhost:5173` i przekazuje `/api` do `http://localhost:8000`. Gdy Compose juz zajmuje port `5173`, zatrzymaj usluge `frontend` albo uruchom `npm run dev -- --port 5174`.

Widok `#/corpora` to galeria korpusow z dwoma kafelkami w rzedzie (jednym na
malym ekranie), nazwa, opisem oraz liczba przykladow (i propozycji, jesli API
udostepnia ich liczbe). Przycisk
`Nowy korpus` znajduje sie nad galeria, bez panelu bocznego.
Po otwarciu korpusu lewy panel zawiera historie czatow i przycisk `Nowy czat`,
srodek zachowuje narzedzia korpusu, a prawy panel sluzy do rozmowy z asystentem.
Rozmowy sa zapisywane w PostgreSQL (tabela `agent_conversations`, migracja
`004`). UUID rozmowy jest nazwa jej folderu w `artifacts/sandbox` i fragmentem
adresu `#/corpora/<korpus>/chats/<rozmowa>`, wiec rozmowe mozna otworzyc z URL.
Nowy czat trafia do bazy przy pierwszej wiadomosci albo zalaczniku. Usuniecie
rozmowy z listy usuwa tez jej folder sandboksa; usuniecie korpusu przenosi jego
rozmowy do kosza (foldery zostaja, wiec przywrocenie korpusu je odzyskuje).
Starsze foldery sandboksa z przypisanym korpusem pojawiaja sie na liscie jako
rozmowy bez historii, a rozmowy zapisane wczesniej w przegladarce sa jednorazowo
przenoszone do bazy.

### Asystent korpusu: orkiestrator, generatory, analityk

Asystent w czacie jest orkiestratorem: rozmawia, ustala intencje uzytkownika,
planuje i zleca prace wykonawcom, a sam robi tylko drobne poprawki. Narzedzia
maja poziomy:

- agenci (top): orkiestrator, `generate_examples` (zespol generatorow, do 5
  rownolegle) i `analyze_series` (analityk serii),
- domenowe (medium): narzedzia korpusu i propozycji oraz czytelnik duzych plikow,
- ogolne (low): sandbox i pliki, dostepne dla kazdego agenta,
- prywatne: `save_examples` generatora oraz `remove_proposals` i
  `regenerate_proposals` analityka; orkiestrator ich nie widzi.

Orkiestrator prowadzi stan sprawy (`update_plan`: intencja, plan krokow,
ustalenia, decyzje, otwarte pytania, serie). Stan jest zapisany przy rozmowie
(`agent_conversations.state`, migracja `005`) i wraca do niego w kazdej turze;
serie i odhaczanie krokow generacji i analizy zapisuje system. Kolejnosc pracy:
zrozumienie (odczyt plikow, przeglad korpusu, intencja), generatory, analityk,
raport. Generator dostaje intencje, cel, tryb treningu (SFT albo DPO), kontekst,
opcje, wytyczne i dane (zakresy plikow do przeczytania albo material), sam czyta
zrodla i zapisuje przyklady do Propozycji paczkami po 20-30 (jedna partia na
serie). W trybie DPO kazdy przyklad ma tez odpowiedz odrzucona
(`metadata.rejected`), widoczna w zakladce Pary DPO i eksporcie DPO. Analityk
usuwa duplikaty i przyklady nienadajace sie do korpusu oraz sam zleca
generatorom regeneracje i runy balansujace. Orkiestrator dostaje tylko skrot:
ile dodano lub usunieto, proporcje i manifest. Wskazane przyklady asystent
pobiera po id (`get_examples`) i edytuje bezposrednio: zaakceptowane
`update_examples` (poprzednia wersja trafia do `data/revisions`, nad lista jest
"Cofnij"), oczekujace propozycje `update_proposals`. Prompty sa w
`prompts/orkiestrator.yml` i `prompts/generator.yml`.

Trening QLoRA jest osobnym profilem z dostepem do GPU:

```powershell
docker compose --profile training run --rm trainer
```

Migracje znajduja sie w `database/changelog` i sa wykonywane automatycznie przez Liquibase przed startem API.

## Wymagania

- Python 3.10+ i CUDA dla treningu.
- Ollama z lokalnym modelem Bielika dla komendy `serve`.
- Konto Hugging Face z zaakceptowanymi warunkami dostepu do modelu oraz `HF_TOKEN` dla treningu.

## Instalacja

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[qlora,dev]"
Copy-Item .env.example .env
```

Ustaw `HF_TOKEN` w zmiennych srodowiskowych przed treningiem. Plik `.env` jest tylko szablonem i nie jest automatycznie ladowany.

## Lokalna inferencja przez Ollama

Profil dla tego komputera (RTX 4080 16 GiB VRAM, 64 GiB RAM) zaklada GGUF `Q4_K_M`: miesci model w calosci na GPU i zostawia zapas dla cache KV. Nie uzywaj Q8, gdy zalezy Ci na kontekscie i plynnosci; wymaga on zbyt malo zapasu VRAM albo offloadu do CPU.

Zamknij dzialajacy proces Ollama, a nastepnie uruchom go z profilem pamieciowym:

```powershell
.\scripts\start-ollama-rtx4080.ps1
```

Skrypt ustawia kontekst 4096 tokenow, cache KV `q8_0`, Flash Attention i pojedynczy rownolegly model. W drugim terminalu zaimportuj lub uruchom lokalny model Bielika:

```powershell
ollama create bielik -f artifacts/Modelfile
$env:OLLAMA_MODEL = "bielik"
bielik-lab serve "Napisz krotkie powitanie po polsku."
```

`artifacts/Modelfile` wskazuje na `artifacts/bielik-q4_k_m.gguf`. Ten plik jest gotowy po uruchomieniu `scripts/convert-bielik-q4.ps1`, ktory konwertuje pobrane Safetensors przez prekompilowany obraz `llama.cpp`. Konwersja wymaga okolo 30 GiB wolnego miejsca na dysku i nie powinna dzialac rownolegle z Ollama ani treningiem.

Czat GUI/API ma temperature `0`, limit wejscia 65000 znakow i limit odpowiedzi 8192 tokenow. Model zachowuje kontekst 32768 tokenow; w praktyce odpowiada to okolo 24 tysiacom tokenow wejscia i 8 tysiacom tokenow wyjscia, z malym zapasem na template ChatML.

Ollama sluzy do inferencji. Adapterow PEFT nie dolacza dynamicznie do endpointu Ollama. Po treningu scal adapter z modelem bazowym, przekonwertuj wynik do GGUF narzedziem `llama.cpp`, a potem utworz lokalny model:

```powershell
bielik-lab adapter merge --base-model speakleash/Bielik-11B-v3.0-Instruct --adapter artifacts/bielik-qlora --output artifacts/bielik-merged
# Konwersja artifacts/bielik-merged do GGUF Q4_K_M przez llama.cpp.
bielik-lab adapter modelfile --gguf artifacts/bielik-merged.gguf
ollama create bielik-adapted -f artifacts/Modelfile
$env:OLLAMA_MODEL = "bielik-adapted"
```

## Korpus

Rekordy maja format JSONL z kluczem `messages`; ostatnia wiadomosc musi nalezec do roli `assistant`.

```powershell
bielik-lab corpus init
bielik-lab corpus import data/examples/support.jsonl --split train
bielik-lab corpus export data/exports/train.jsonl --split train
bielik-lab corpus export data/exports/validation.jsonl --split validation
```

## Trening i walidacja

`configs/qlora.yaml` jest profilem dla RTX 4080: 4-bitowe NF4, podwojna kwantyzacja, batch 1, sekwencje 1024, checkpointing gradientow i limit 14 GiB VRAM. Pozostawia to margines dla systemu i nie korzysta z cache KV podczas treningu. `configs/lora.yaml` dla LoRA bez kwantyzacji wymaga znacznie wiekszego VRAM i nie jest zalecany na tej karcie.

```powershell
bielik-lab train --config configs/qlora.yaml
bielik-lab evaluate --config configs/qlora.yaml
```

Przed treningiem wyeksportuj oba splity: `train` i `validation`. Skonfiguruj rozmiary batcha, akumulacje gradientu i maksymalna dlugosc sekwencji odpowiednio do VRAM.
