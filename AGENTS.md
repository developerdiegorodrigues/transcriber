# AGENTS.md

Orientações para qualquer agente que trabalhe neste repositório. Leia também
[TODO.md](TODO.md) (o que fazer, em ordem) e
[reference/Progress_Control.md](reference/Progress_Control.md) (o que já foi medido —
não reinvestigue hardware antes de consultá-lo).

## O que é o projeto

CLI de transcrição local de vídeos com `faster-whisper` (backend principal) e
openai-whisper (backend legado, via subprocesso). Nada de servidor, nada de nuvem:
tudo roda na máquina do usuário, com GPU quando disponível.

## Ambiente

Máquina Linux com NVIDIA RTX 5060 Ti (Blackwell, `sm_120`, 16 GiB), driver 595,
CUDA 13.2, Python 3.12.

**A GPU é compartilhada.** Outras cargas de IA rodam em paralelo nesta máquina e
frequentemente ocupam 13 GiB dos 16 GiB. Duas consequências:

- Nunca aloque VRAM de forma generosa em testes exploratórios. Antes de carregar
  qualquer modelo, verifique `torch.cuda.mem_get_info()`; se houver pouca memória
  livre, use o modelo `tiny` ou peça para o usuário liberar a placa. Estourar a VRAM
  pode derrubar um trabalho dele que estava em andamento.
- Qualquer medição de desempenho feita com a GPU disputada é inválida como linha de
  base. Registre a condição junto do número ou não registre o número.

Há um conflito de linhas CUDA neste ambiente: o CTranslate2 é compilado contra CUDA 12
e o PyTorch traz a linha CUDA 13. Está resolvido — `cuda_runtime.py` pré-carrega
`libcublas.so.12` e `libcudnn.so.9` por caminho absoluto, e `nvidia-cublas-cu12` está
fixado no lockfile. Não remova nenhum dos dois: sem isso o backend `faster-whisper`
não roda em GPU, e a sonda do driver sozinha não detecta a falha (descobertas D1–D3).

## Instalação e execução

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
```

```bash
.venv/bin/transcriber doctor            # diagnóstico; use antes de culpar o código
.venv/bin/transcriber video.mp4 --profile fast
.venv/bin/transcriber benchmark video.mp4 --profiles fast quality
```

## Testes

Apenas biblioteca padrão, sem dependências de teste e sem rede:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Todo teste roda em máquina sem GPU. Acesso a CUDA, a subprocessos (`ffmpeg`, `demucs`,
`whisper`) e ao sistema de arquivos externo entra por `unittest.mock.patch`. Não
introduza `pytest`, `numpy` ou fixtures binárias nos testes.

## Convenções de código

- **Código e docstrings em inglês; mensagens ao usuário em português.** Essa separação
  é consistente em todo o repositório — mantenha-a.
- `from __future__ import annotations` no topo de todo módulo.
- Dataclasses de configuração e de resultado são `frozen=True, slots=True`. Para
  derivar uma variante, use `dataclasses.replace`, não mutação.
- Validação de configuração vive em `TranscriptionConfig.__post_init__`, não espalhada
  pelo CLI.
- Erros esperados herdam de `TranscriberError` (`HardwareError`, `DependencyError`,
  `TranscriptionError`). O `main()` os converte em `ERRO: ...` e código 1. Não deixe
  rastreamento de pilha chegar ao usuário por falha prevista.
- Importações pesadas (`torch`, `ctranslate2`, `faster_whisper`, `whisper`, `demucs`)
  são locais à função, nunca no topo do módulo. Isso mantém `doctor`, `--help` e os
  testes rápidos e funcionais sem GPU.
- Sem emojis e sem cores ANSI na saída.

## Invariantes que não podem ser quebradas

1. **Nada de instalar ou atualizar pacotes em tempo de execução.** O ambiente é
   preparado uma vez pelo `requirements.lock`. Se uma dependência falta, é
   `DependencyError` com instrução, não `pip install`.
2. **Publicação atômica.** Artefatos são produzidos em diretório temporário no mesmo
   sistema de arquivos e movidos para `output/<nome>/` só após sucesso completo. Uma
   saída existente nunca é sobrescrita implicitamente: exige `--force` (que ainda
   mantém backup até a troca terminar) ou `--skip-existing`.
3. **Sem fallback silencioso de dispositivo.** `--device cuda` sem CUDA é erro;
   `--device auto` cai para CPU com aviso explícito. Nunca troque de dispositivo,
   modelo ou precisão sem dizer ao usuário.
4. **Opção explícita vence perfil.** Qualquer `--flag` passada na linha de comando
   sobrescreve o valor do perfil, sempre.
5. **Sem caminhos pessoais nos relatórios.** `doctor --json`, `metadata.json` e os
   relatórios de benchmark são colados em issues: não registram caminho absoluto da
   mídia nem do diretório de saída. Há teste garantindo isso.
6. **Áudio intermediário é temporário.** O perfil `lyrics` cria WAV apenas dentro de um
   diretório temporário exclusivo, removido mesmo em caso de falha ou `Ctrl+C`.
7. **`Ctrl+C` retorna 130**, limpa temporários e preserva o que já foi publicado.

## Armadilhas do backend já pagas

Três comportamentos do faster-whisper 1.2.1 que custaram tempo para descobrir e estão
cobertos por teste. Não os desfaça sem medir:

1. **A inferência em lote exige fronteiras de segmento**, que normalmente vêm do VAD.
   Com `vad_filter=False` e batch > 1, o backend recusa qualquer áudio acima de 30 s.
   Os perfis `quality` e `lyrics` caem para batch 1 por isso; `--window-batching`
   contorna cobrindo o áudio com janelas.
2. **Um `clip_timestamps` maior que 30 s é truncado em silêncio** — só os primeiros
   30 s de cada clipe são transcritos, com um aviso no log e nada mais. Por isso
   `_window_clips` nunca gera janelas maiores que `WINDOW_SECONDS`.
3. **O pipeline em lote descarta tokens de timestamp por padrão**, colapsando cada
   chunk em um bloco de legenda. O projeto passa `without_timestamps=False`
   explicitamente nos dois caminhos.

Os testes mockam o pipeline, então nenhum deles pega esses casos sozinho — eles foram
encontrados rodando mídia real. Ao mexer em inferência, rode uma transcrição de
verdade, com mídia acima de 30 s, e confira o `.srt`, não só o `.txt`.

## Medições

Toda medição de desempenho exige GPU ociosa **e memória livre**: um processo vizinho
pode reter 12 GiB mesmo com 1 % de utilização, e `nvidia-smi --query-compute-apps`
revela isso. Além disso, as mídias deste diretório são musicais e o VAD detecta de 1,8 % a
7,7 % de fala, então qualquer ×RT medido com VAD ligado mede o VAD. Meça com o perfil
`quality` (VAD desligado) para que a carga seja real.

## Métricas que enganam

- **A contagem de segmentos suspeitos (`low_confidence_segments`) varia ±27 % entre
  execuções idênticas**, porque os sinais de qualidade são comparações com limiar e os
  valores perto do limiar oscilam por variação numérica da GPU — mesmo com o texto
  saindo igual. Nunca compare duas configurações por uma execução só; já produziu uma
  conclusão errada neste projeto (D15/D17).
- **Divergência de texto (WER contra outra saída) não é acurácia.** O modo sequencial
  alucina laços de repetição em música, então "mais palavras" ou "mais diferente" não
  diz qual saída é melhor. Sem referência humana, relate a divergência como diferença,
  nunca como erro.
- **×RT com VAD ligado em mídia musical mede o VAD**, que descarta de 95 % a 99 % desse
  áudio.

## Mapa do código

| Arquivo | Responsabilidade |
| --- | --- |
| `cli.py` | Argumentos, despacho de subcomandos, tradução de erros em código de saída |
| `config.py` | `Profile`, `TranscriptionConfig`, perfis prontos, validação |
| `hardware.py` | Sondas de CUDA, resolução de dispositivo, relatório do `doctor`, snapshot de hardware |
| `cuda_runtime.py` | Pré-carga das bibliotecas CUDA 12 que o CTranslate2 exige |
| `vram.py` | Orçamento de VRAM: batch inicial e diagnóstico de falta de memória |
| `pipeline.py` | Despacho por backend e orquestração da separação vocal |
| `jobs.py` | Execução segura, `metadata.json`, publicação atômica |
| `backends/faster_whisper.py` | Inferência em processo, recuo de batch, reavaliação de trechos |
| `backends/openai_whisper.py` | Adaptador para o CLI do whisper |
| `separation.py` | Demucs em subprocesso, em diretório temporário |
| `quality.py` | Marcação de baixa confiança e relatórios de revisão |
| `benchmark.py` | Medição de tempo, VRAM e WER |
| `outputs.py` / `ui.py` / `media.py` | Escrita de formatos, seletor interativo, descoberta de mídia |

`transcribe.py` na raiz é entrada compatível para uso direto do checkout.

## Ao trabalhar nas melhorias de hardware

- Siga a ordem das fases do `TODO.md`. A Fase 0 destrava o caminho GPU; sem ela,
  medições de batch e precisão não têm significado.
- Ao concluir um item, marque a caixa no `TODO.md` **e** registre a descoberta ou a
  medição em `reference/Progress_Control.md`, com a condição de medição. Esses dois
  arquivos são o estado do projeto entre sessões.
- Não altere defaults de perfil (batch, precisão, modelo) sem benchmark com a GPU
  ociosa. "Deve ser mais rápido" não é evidência. As pendências que restam no TODO
  estão bloqueadas exatamente por isso, não por falta de decisão.
- Ao validar algo na GPU, prefira `doctor --smoke-test` e o modelo `tiny`: custam
  pouca VRAM e não competem com o trabalho do usuário.

## Git

Não faça commit nem push sem o usuário pedir. A branch principal é `main`; mensagens de
commit em português, no formato `feat: ...` / `fix: ...`, como o histórico existente.
