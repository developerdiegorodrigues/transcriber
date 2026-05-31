# transcribe

Ferramenta de linha de comando para **transcrição de vídeos** usando o [OpenAI Whisper](https://github.com/openai/whisper).

O script exibe uma lista interativa (TUI) dos vídeos presentes no diretório, converte o arquivo selecionado para WAV via `ffmpeg` e gera a transcrição com o Whisper em múltiplos formatos.

## Recursos

- 🎛️ **Interface interativa no terminal** (curses) para selecionar o vídeo.
- 🔄 **Conversão automática** de vídeo para áudio WAV (16 kHz, mono) com `ffmpeg`.
- 🧠 **Transcrição com Whisper** (modelo `large` por padrão).
- 📦 **Auto-instalação**: cria automaticamente um ambiente virtual `.venv` e instala o `openai-whisper`.
- 📁 **Múltiplos formatos de saída** (`txt`, `srt`, `vtt`, `json`, `tsv`) salvos em `output/`.
- 🔁 Permite transcrever **vários arquivos** em sequência.

## Imagens

<img align=right width=680 src=https://github.com/developerdiegorodrigues/transcribe/blob/main/images/converter_680x336.png />

<img align=right width=345 src=https://github.com/developerdiegorodrigues/transcribe/blob/main/images/converter_345x472.png />

## Formatos de vídeo suportados

`.mp4` · `.mkv` · `.avi` · `.mov` · `.webm` · `.flv` · `.wmv` · `.m4v`

<br/>
<br/>
<br/>

## Pré-requisitos

- **Python 3** (com o módulo `venv`)
- **ffmpeg** instalado no sistema:

  ```bash
  sudo apt install ffmpeg
  ```

> O pacote `openai-whisper` é instalado automaticamente pelo próprio script em um ambiente virtual `.venv`.

## Uso

1. Coloque os arquivos de vídeo na mesma pasta do `transcribe.py`.
2. Execute:

   ```bash
   python3 transcribe.py
   ```

3. Na primeira execução, o script cria o `.venv` e instala as dependências automaticamente, reiniciando em seguida dentro do ambiente virtual.
4. Use as teclas para navegar e selecionar o vídeo:

   | Tecla            | Ação                  |
   | ---------------- | --------------------- |
   | `↑` / `k`        | Mover para cima       |
   | `↓` / `j`        | Mover para baixo      |
   | `Enter`          | Selecionar o arquivo  |
   | `q` / `Esc`      | Sair                  |

5. Ao final, o script pergunta se você deseja transcrever outro arquivo.

## Saída

As transcrições são salvas em:

```
output/<nome-do-video>/
```

Um arquivo temporário `output/audio.wav` é gerado durante a conversão e removido automaticamente após a transcrição.

## Configuração

Os parâmetros do Whisper podem ser ajustados diretamente no início do `transcribe.py`:

```python
WHISPER_MODEL    = "large"     # tiny, base, small, medium, large
WHISPER_LANGUAGE = "English"   # idioma do áudio
WHISPER_DEVICE   = "cpu"       # cpu ou cuda (GPU)
```

> 💡 O modelo `large` em CPU pode levar vários minutos. Para acelerar, use um modelo menor ou um dispositivo com GPU (`cuda`).

## Estrutura do projeto

```
.
├── transcribe.py   # script principal
├── output/         # transcrições geradas
├── LICENSE
└── README.md
```

## Licença

Este projeto está licenciado sob os termos do arquivo [LICENSE](LICENSE).