# transcribe

Ferramenta de linha de comando para transcrição local de vídeos com
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) e
[OpenAI Whisper](https://github.com/openai/whisper). Ela oferece perfis de execução,
seleção interativa, processamento em lote, detecção automática de CUDA, benchmark e
saídas em múltiplos formatos.

## Recursos

- Interface interativa no terminal para selecionar vídeos.
- Entrada direta de mídia, sem criar um WAV intermediário.
- Seleção automática entre CUDA e CPU.
- Backend `faster-whisper` com batch adaptativo em caso de falta de VRAM.
- Perfis prontos para velocidade, qualidade, CPU e compatibilidade.
- Comando de diagnóstico para driver, CUDA, PyTorch, Whisper e FFmpeg.
- Configuração por linha de comando de modelo, idioma, dispositivo e formato.
- Saídas `txt`, `srt`, `vtt`, `json` e `tsv` organizadas em `output/`.
- Dependências fixadas para instalações reproduzíveis.
- Benchmark de velocidade, pico de VRAM e WER opcional.

## Pré-requisitos

- Python 3.10 ou mais recente, com suporte a `venv`.
- FFmpeg instalado no sistema.
- Para o backend OpenAI em GPU, driver NVIDIA compatível com o PyTorch instalado.
- Para `faster-whisper` em GPU, CUDA 12 com cuBLAS e cuDNN 9.

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

O lockfile instala o CTranslate2, mas não força bibliotecas CUDA 12 dentro do
virtualenv para evitar conflito com o runtime CUDA do PyTorch. Em uma máquina com
NVIDIA, instale CUDA 12/cuDNN 9 no sistema e confirme o resultado com `doctor`.

## Diagnóstico

Antes da primeira transcrição — e novamente depois de trocar a placa ou o driver —
execute:

```bash
.venv/bin/transcriber doctor
```

O diagnóstico mostra PyTorch e CTranslate2 separadamente, pois um backend pode ter
acesso à GPU enquanto o outro não. Também informa o resultado de `nvidia-smi` e o
dispositivo automático de cada backend.

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

## Perfis

| Perfil | Backend | Modelo | Dispositivo | Precisão | Batch inicial |
| --- | --- | --- | --- | --- | ---: |
| `fast` (padrão) | faster-whisper | `turbo` | auto | FP16 CUDA / INT8 CPU | 8 |
| `quality` | faster-whisper | `large-v3` | auto | FP16 CUDA / INT8 CPU | 4 |
| `cpu` | faster-whisper | `small` | CPU | INT8 | 4 |
| `legacy` | openai-whisper | `large` | auto | FP16 CUDA / FP32 CPU | 1 |

Se o CTranslate2 ficar sem VRAM, o batch é reduzido sucessivamente até 1. Em GPUs
Blackwell/RTX 50, use FP16: versões atuais do CTranslate2 desabilitam INT8 nessa
arquitetura.

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
WER quando disponível. O download dos modelos é preparado antes do cronômetro; o
tempo medido inclui o carregamento e a inferência. Os artefatos e relatórios de
benchmark não são versionados.

## Saída

Cada mídia recebe seu próprio diretório e preserva o nome original nos artefatos:

```text
output/<nome-do-video>/<nome-do-video>.txt
output/<nome-do-video>/<nome-do-video>.srt
...
```

O Whisper recebe o vídeo diretamente e faz internamente a leitura da faixa de
áudio. O antigo `output/audio.wav` temporário não é mais necessário.

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
│   ├── hardware.py
│   ├── media.py
│   ├── outputs.py
│   ├── pipeline.py
│   └── ui.py
├── tests/
└── transcribe.py
```

## Imagens

<img width="680" src="https://github.com/developerdiegorodrigues/transcribe/blob/main/images/converter_680x336.png" alt="Seletor interativo de vídeos" />

<img width="345" src="https://github.com/developerdiegorodrigues/transcribe/blob/main/images/converter_345x472.png" alt="Progresso de uma transcrição" />

## Licença

Este projeto está licenciado sob os termos do arquivo [LICENSE](LICENSE).
