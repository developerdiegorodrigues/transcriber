# transcribe

Ferramenta de linha de comando para transcrição local de vídeos com
[OpenAI Whisper](https://github.com/openai/whisper). Ela oferece seleção interativa,
processamento em lote, detecção automática de CUDA e saídas em múltiplos formatos.

## Recursos

- Interface interativa no terminal para selecionar vídeos.
- Entrada direta de mídia, sem criar um WAV intermediário.
- Seleção automática entre CUDA e CPU.
- Comando de diagnóstico para driver, CUDA, PyTorch, Whisper e FFmpeg.
- Configuração por linha de comando de modelo, idioma, dispositivo e formato.
- Saídas `txt`, `srt`, `vtt`, `json` e `tsv` organizadas em `output/`.
- Dependências fixadas para instalações reproduzíveis.

## Pré-requisitos

- Python 3.10 ou mais recente, com suporte a `venv`.
- FFmpeg instalado no sistema.
- Para GPU, driver NVIDIA compatível com o runtime CUDA instalado.

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

## Diagnóstico

Antes da primeira transcrição — e novamente depois de trocar a placa ou o driver —
execute:

```bash
.venv/bin/transcriber doctor
```

O diagnóstico mostra a versão do PyTorch, o runtime CUDA, o resultado de
`nvidia-smi`, as GPUs acessíveis e o dispositivo que seria escolhido pelo modo
automático.

## Uso

Abra o seletor interativo para os vídeos do diretório atual:

```bash
.venv/bin/transcriber
```

Transcreva um ou vários arquivos diretamente:

```bash
.venv/bin/transcriber video.mp4
.venv/bin/transcriber primeiro.mp4 segundo.mkv
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
--model NOME
--language IDIOMA|auto
--output-dir DIRETÓRIO
--output-format all|txt|vtt|srt|tsv|json
```

Exemplos:

```bash
.venv/bin/transcriber video.mp4 --device cuda --model large-v3
.venv/bin/transcriber video.mp4 --language auto --output-format srt
```

Com `--device auto`, CUDA é usada quando estiver realmente acessível pelo PyTorch;
caso contrário, a CPU é usada com um aviso. Com `--device cuda`, a ausência de CUDA
é tratada como erro, evitando um fallback silencioso.

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
│   ├── cli.py
│   ├── config.py
│   ├── hardware.py
│   ├── media.py
│   └── ui.py
├── tests/
└── transcribe.py
```

## Imagens

<img width="680" src="https://github.com/developerdiegorodrigues/transcribe/blob/main/images/converter_680x336.png" alt="Seletor interativo de vídeos" />

<img width="345" src="https://github.com/developerdiegorodrigues/transcribe/blob/main/images/converter_345x472.png" alt="Progresso de uma transcrição" />

## Licença

Este projeto está licenciado sob os termos do arquivo [LICENSE](LICENSE).
