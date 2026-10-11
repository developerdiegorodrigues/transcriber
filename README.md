# transcribe

Ferramenta de linha de comando para transcrição local de vídeos com
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) e
[OpenAI Whisper](https://github.com/openai/whisper). Ela oferece perfis de execução,
seleção interativa, processamento em lote, detecção automática de CUDA, benchmark e
saídas em múltiplos formatos.

## Recursos

- Interface interativa no terminal para selecionar vídeos.
- Entrada direta de mídia nos perfis gerais, sem WAV intermediário persistente.
- Seleção automática entre CUDA e CPU.
- Backend `faster-whisper` com batch dimensionado pela VRAM livre, para cima ou
  para baixo, e recuo adicional em caso de falta de memória.
- Perfis prontos para velocidade, qualidade, CPU e compatibilidade.
- Perfil para letras com separação vocal Demucs e revisão de baixa confiança.
- Seleção múltipla na interface interativa.
- Publicação atômica, retomada por arquivo e metadados de execução.
- Comando de diagnóstico para driver, CUDA, PyTorch, Whisper e FFmpeg.
- Configuração por linha de comando de modelo, idioma, dispositivo e formato.
- Saídas `txt`, `srt`, `vtt`, `json` e `tsv` organizadas em `output/`.
- Dependências fixadas para instalações reproduzíveis.
- Benchmark de velocidade, pico de VRAM e WER opcional.

## Pré-requisitos

- Python 3.10 ou mais recente, com suporte a `venv`.
- FFmpeg instalado no sistema.
- Para o backend OpenAI em GPU, driver NVIDIA compatível com o PyTorch instalado.
- Para `faster-whisper` em GPU, cuBLAS da linha CUDA 12 e cuDNN 9. O lockfile já
  instala `nvidia-cublas-cu12`; nada precisa ser instalado no sistema.

No Ubuntu, instale os pacotes de sistema com:

```bash
sudo apt install python3-venv ffmpeg
```

## Instalação reproduzível

O programa não instala nem atualiza pacotes durante a execução. Prepare o ambiente
uma única vez:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
```

O `requirements.lock` representa o ambiente Linux/CUDA validado pelo projeto. O
`pyproject.toml` contém os metadados do pacote e disponibiliza o comando
`.venv/bin/transcriber`.

O CTranslate2 é compilado contra CUDA 12 e resolve `libcublas.so.12` pelo carregador
do sistema, enquanto o PyTorch traz a própria linha CUDA 13 e resolve a dela
internamente. As duas convivem no mesmo virtualenv porque os *sonames* são distintos,
e o lockfile fixa as duas: sem `nvidia-cublas-cu12`, o `faster-whisper` não roda em
GPU mesmo com driver e placa perfeitos.

O programa pré-carrega essas bibliotecas por caminho absoluto antes de usar o
CTranslate2, então nenhum `LD_LIBRARY_PATH` é necessário. Confirme o resultado com
`doctor`.

## Diagnóstico

Antes da primeira transcrição — e novamente depois de trocar a placa ou o driver —
execute:

```bash
.venv/bin/transcriber doctor
```

O diagnóstico mostra PyTorch e CTranslate2 separadamente, pois um backend pode ter
acesso à GPU enquanto o outro não. Também informa o resultado de `nvidia-smi` e o
dispositivo automático de cada backend.

Para gerar um relatório estruturado que possa ser comparado ou anexado a uma issue,
sem expor caminhos pessoais do sistema:

```bash
.venv/bin/transcriber doctor --json
```

### Validação após o upgrade da GPU

Depois de trocar a placa ou o driver, valide separadamente os dois backends. Cada
comando termina com código diferente de zero se a GPU não estiver acessível, o que
permite usá-los também em scripts:

```bash
.venv/bin/transcriber doctor --require-cuda --backend faster-whisper --smoke-test
.venv/bin/transcriber doctor --require-cuda --backend openai-whisper
```

O `--smoke-test` carrega um modelo pequeno na GPU e transcreve um segundo de áudio
sintético. Ele existe porque a detecção de dispositivo não é prova de funcionamento: o
driver pode expor a placa ao CTranslate2 enquanto as bibliotecas que ele exige não
carregam, e nesse caso a transcrição falharia apesar de todo diagnóstico indicar
`sim`. Quando isso acontece, o `doctor` aponta a biblioteca ausente e falha.

Os dois resultados devem mostrar a placa com a VRAM esperada, e as linhas
`CUDA disponível`, `CUDA no CTranslate2` e `Bibliotecas CUDA do CTranslate2` devem
indicar, respectivamente, `sim`, `sim` e `completas`. O diagnóstico também informa a
compute capability, as arquiteturas compiladas no PyTorch e as precisões que o
CTranslate2 aceita nesta placa — é por essa lista que se descobre, por exemplo, se
INT8 está disponível.

Depois disso, registre uma linha de base com a mesma mídia, **com a GPU ociosa**,
antes de alterar modelos ou batches:

```bash
.venv/bin/transcriber benchmark video.mp4 --profiles fast quality
```

## Uso

Abra o seletor interativo para os vídeos do diretório atual:

```bash
.venv/bin/transcriber
```

Transcreva um ou vários arquivos diretamente:

```bash
.venv/bin/transcriber video.mp4
.venv/bin/transcriber primeiro.mp4 segundo.mkv --profile fast
```

O subcomando explícito também é aceito:

```bash
.venv/bin/transcriber transcribe video.mp4
```

O arquivo `transcribe.py` permanece como entrada compatível para uso a partir do
checkout:

```bash
.venv/bin/python transcribe.py video.mp4
```

### Opções principais

```text
--device auto|cpu|cuda
--profile fast|quality|cpu|legacy
--backend faster-whisper|openai-whisper
--model NOME
--language IDIOMA|auto
--output-dir DIRETÓRIO
--output-format all|txt|vtt|srt|tsv|json
--compute-type TIPO
--batch-size N
--beam-size N
--vad|--no-vad
--word-timestamps|--no-word-timestamps
--window-batching|--no-window-batching
--separate-vocals|--no-separate-vocals
--separation-device auto|cpu|cuda
--quality-review|--no-quality-review
--retry-low-confidence|--no-retry-low-confidence
--skip-existing
--force
```

Exemplos:

```bash
.venv/bin/transcriber video.mp4 --profile quality
.venv/bin/transcriber video.mp4 --device cuda --model large-v3
.venv/bin/transcriber video.mp4 --language auto --output-format srt
```

Com `--device auto`, CUDA é usada quando estiver realmente acessível pelo PyTorch;
caso contrário, a CPU é usada com um aviso. Com `--device cuda`, a ausência de CUDA
é tratada como erro, evitando um fallback silencioso. Qualquer opção explícita
sobrescreve o valor definido pelo perfil.

Na interface interativa, use `Espaço` para marcar vários vídeos, `a` para marcar ou
desmarcar todos e `Enter` para confirmar. Sem itens marcados, `Enter` processa apenas
o item sob o cursor.

## Perfis

| Perfil | Backend | Modelo | Dispositivo | Precisão | Batch inicial |
| --- | --- | --- | --- | --- | ---: |
| `fast` (padrão) | faster-whisper | `turbo` | auto | FP16 CUDA / INT8 CPU | 8 |
| `quality` | faster-whisper | `large-v3` | auto | FP16 CUDA / INT8 CPU | 4 |
| `cpu` | faster-whisper | `small` | CPU | INT8 | 4 |
| `legacy` | openai-whisper | `large` | auto | FP16 CUDA / FP32 CPU | 1 |
| `lyrics` | faster-whisper + Demucs | `large-v3` | auto | FP16 CUDA / INT8 CPU | 4 |

O batch da tabela é um ponto de partida: quando não há `--batch-size` explícito, ele
é reduzido conforme a VRAM **livre** no momento da execução, para que o trabalho caiba
quando outro processo ocupa a placa. Se o CTranslate2 ainda ficar sem memória durante a
inferência, o batch é reduzido sucessivamente até 1.

Ele nunca é aumentado, e isso é medido, não conservadorismo: numa RTX 5060 Ti ociosa, o
`large-v3` levou 26,8 s em batch 4 e 54,3 s em batch 32 sobre os mesmos 1120 s de
áudio, e o `turbo` em batch 32 ficou mais lento que em batch 1 — apesar de o pico de
VRAM (12,6 GiB) caber folgado nos 16 GiB. VRAM livre compra um modelo maior, não um
batch maior.

### Quando o VAD descarta a mídia

O filtro de voz (VAD) do perfil `fast` classifica vocais cantados sobre instrumental
como não-fala. Numa música, isso descarta quase tudo: medido em dois vídeos deste
repositório, o VAD manteve 4 % e 1 % do áudio, e a transcrição saiu com 23 e 161
palavras. O programa avisa quando a cobertura cai abaixo de metade do áudio:

```text
Aviso: o VAD manteve apenas 12s de 280s (4% do áudio); o restante não foi transcrito.
```

Dois sintomas acompanham essa perda. O texto some, e as legendas que sobram ficam
desalinhadas, porque os vãos descartados são absorvidos no fim de cada bloco — um
trecho de três palavras pode ocupar 36 segundos de legenda.

Para conteúdo musical, use `--profile lyrics` (isola o vocal com Demucs e não usa VAD)
ou `--no-vad`. Na mesma música, o `fast` produziu 23 palavras até 00:53 e o `lyrics`
produziu 379 cobrindo os 4m40s.

### Inferência em lote e VAD

A inferência em lote do faster-whisper obtém as fronteiras de segmento do VAD. Os
perfis `quality` e `lyrics` desligam o VAD de propósito, para não descartar áudio
cantado ou baixo — e nessa combinação o backend recusa qualquer mídia acima de 30
segundos. Nesses perfis o programa usa batch 1 e diz por quê.

Com `--window-batching`, o áudio é coberto por janelas de no máximo 30 segundos e o
lote volta a funcionar. No MAVIRA (280 s, `large-v3`, FP16, GPU ociosa), o perfil
`quality` caiu de 27,09 s para 7,06 s — 3,8× mais rápido, com batch automático 32.

A contrapartida é de acurácia e não é desprezível: o texto divergiu 31 % do modo
sequencial na mesma mídia. Parte dessa diferença é o modo sequencial repetindo trechos
(alucinação de laço, comum em música), então nenhum dos dois é a referência correta.
Enquanto não houver uma transcrição humana para comparar, o padrão é o modo seguro e o
ganho é opcional.

A precisão padrão em GPU é FP16. As precisões realmente disponíveis dependem da
arquitetura e são listadas por `doctor`; em Blackwell/RTX 50 o CTranslate2 4.8
também aceita `int8_float16` e `bfloat16`, úteis quando a VRAM está disputada:

```bash
.venv/bin/transcriber video.mp4 --compute-type int8_float16
```

Se a VRAM livre não acomodar o modelo, o erro informa quanto havia disponível, quanto
o modelo exige e as alternativas, em vez de apenas `out of memory`.

## Transcrição de letras

O perfil `lyrics` foi desenhado para vídeos musicais:

```bash
.venv/bin/transcriber musica.mp4 --profile lyrics --language English
```

O fluxo desse perfil é:

1. O FFmpeg extrai temporariamente áudio estéreo em 44,1 kHz.
2. O Demucs `htdemucs` isola o stem vocal.
3. O processo do Demucs termina, liberando CPU/GPU antes de carregar o Whisper.
4. O `large-v3` transcreve os vocais sem VAD e sem carregar contexto entre janelas.
5. Intervalos de baixa confiança são agrupados e reavaliados com temperatura zero.
6. A segunda passagem só substitui um intervalo quando melhora a probabilidade média.
7. Todo áudio intermediário é removido mesmo se a execução falhar.

Além dos formatos normais, são gerados:

```text
<nome>.review.json
<nome>.review.txt
```

Esses relatórios apontam baixa probabilidade, compressão excessiva, alta probabilidade
de silêncio ou temperatura elevada. Eles servem como fila objetiva de revisão; o
texto suspeito não é apagado silenciosamente.

A separação padrão é o `htdemucs`. O `htdemucs_ft`, ensemble de quatro modelos, está
disponível e não custa VRAM adicional (4172 MiB contra 4140 MiB, pois os modelos rodam
em sequência), mas custa 2,4× no tempo — 66,7 s contra 27,3 s numa música de 4m40s — e
não demonstrou ganho de qualidade acima do ruído de medição nos testes feitos:

```bash
.venv/bin/transcriber musica.mp4 --profile lyrics --demucs-model htdemucs_ft
```

As opções do perfil podem ser desativadas ou ajustadas individualmente:

```bash
.venv/bin/transcriber musica.mp4 --profile lyrics --no-separate-vocals
.venv/bin/transcriber musica.mp4 --profile lyrics --separation-device cpu
.venv/bin/transcriber musica.mp4 --profile quality --quality-review
```

## Benchmark

Compare os perfis usando a mesma mídia:

```bash
.venv/bin/transcriber benchmark video.mp4 --profiles fast quality cpu
```

Para também calcular Word Error Rate (WER), forneça uma transcrição de referência:

```bash
.venv/bin/transcriber benchmark video.mp4 \
  --profiles fast quality \
  --reference referencia.txt
```

O relatório JSON é salvo em `benchmarks/results/` e contém tempo total, fator de
tempo real, áudio processado por segundo, batch efetivo, precisão, pico de VRAM e
WER quando disponível. Um bloco `hardware` identifica GPU, driver, compute
capability, VRAM e versões, e cada resultado registra a VRAM livre no início e
quantos outros processos disputavam a placa — sem isso, dois relatórios de gerações
diferentes não são comparáveis, e uma medição feita durante outra carga parece
regressão. O download dos modelos é preparado antes do cronômetro; o
tempo medido inclui o carregamento e a inferência. Os artefatos e relatórios de
benchmark não são versionados.

## Saída

Cada mídia recebe seu próprio diretório e preserva o nome original nos artefatos:

```text
output/<nome-do-video>/<nome-do-video>.txt
output/<nome-do-video>/<nome-do-video>.srt
output/<nome-do-video>/<nome-do-video>.metadata.json
...
```

Nos perfis gerais, o Whisper recebe o vídeo diretamente. O perfil `lyrics` cria
áudio apenas dentro de um diretório temporário exclusivo, removido ao fim da etapa;
o antigo `output/audio.wav` compartilhado não é mais necessário.

Os artefatos são produzidos primeiro em um diretório temporário no mesmo sistema de
arquivos e publicados juntos apenas após o sucesso. Uma saída existente nunca é
sobrescrita implicitamente:

```bash
# Ignora itens já processados em um lote
.venv/bin/transcriber *.mp4 --skip-existing

# Substitui a saída somente depois que a nova transcrição terminar
.venv/bin/transcriber video.mp4 --force
```

O arquivo `metadata.json` registra perfil, modelo, backend, precisão, batch efetivo,
tempo gasto, versões e lista de artefatos. Ele não registra o caminho absoluto da
mídia ou do diretório de saída. Em `Ctrl+C`, o programa retorna código 130 e remove
os arquivos temporários, preservando qualquer resultado publicado anteriormente.

## Desenvolvimento e testes

Os testes usam apenas a biblioteca padrão:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Estrutura principal:

```text
.
├── pyproject.toml
├── requirements.lock
├── src/transcriber/
│   ├── backends/
│   ├── benchmark.py
│   ├── cli.py
│   ├── config.py
│   ├── cuda_runtime.py
│   ├── hardware.py
│   ├── jobs.py
│   ├── media.py
│   ├── outputs.py
│   ├── pipeline.py
│   ├── quality.py
│   ├── separation.py
│   ├── ui.py
│   └── vram.py
├── tests/
└── transcribe.py
```

## Imagens

<img width="680" src="https://github.com/developerdiegorodrigues/transcribe/blob/main/images/converter_680x336.png" alt="Seletor interativo de vídeos" />

<img width="345" src="https://github.com/developerdiegorodrigues/transcribe/blob/main/images/converter_345x472.png" alt="Progresso de uma transcrição" />

## Licença

Este projeto está licenciado sob os termos do arquivo [LICENSE](LICENSE).
