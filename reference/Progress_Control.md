# Controle de progresso e descobertas

Registro resumido do que foi medido, decidido e concluído neste projeto. Serve como
memória entre sessões: antes de reinvestigar hardware, leia aqui.

Convenção: toda afirmação de desempenho vem acompanhada da condição de medição. Número
sem condição é número inútil.

## Ambiente de referência (medido em 2026-10-10)

| Item | Valor |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16311 MiB |
| Arquitetura | Blackwell, compute capability 12.0 (`sm_120`) |
| Driver | 595 |
| CUDA do driver | 13.2 |
| Sistema | Linux x86_64, Python 3.12 |
| PyTorch | 2.12.0+cu130 (arch list inclui `sm_120`) |
| CTranslate2 | 4.8.2 (última no PyPI) |
| faster-whisper | 1.2.1 (última no PyPI) |
| Demucs | 4.1.0 |
| openai-whisper | 20250625 |

A GPU é compartilhada com outras cargas de IA. No momento da auditoria havia 13,6 GiB
de 16 GiB ocupados por outro processo de inferência, com ~2 GiB livres e 100 % de
utilização.
Isso não é exceção: é o cenário normal desta máquina e precisa ser tratado como tal
pelo código.

## Descobertas

### D1 — `faster-whisper` não executa em GPU neste ambiente (crítico)

O CTranslate2 4.8.2 é compilado contra CUDA 12 e exige `libcublas.so.12`. O
`requirements.lock` instala apenas a linha cu13 (`nvidia-cublas==13.1.1.3`,
`nvidia-cudnn-cu13`, `torch==2.12.0+cu130`), e não existe `libcublas.so.12` no sistema
nem no virtualenv. Os perfis `fast`, `quality` e `lyrics` falham em GPU:

```
RuntimeError: Library libcublas.so.12 is not found or cannot be loaded
```

A falha ocorre após o carregamento do modelo, não na detecção de dispositivo.

### D2 — O `doctor` não detecta D1 (falso positivo)

```
CUDA no CTranslate2: sim
Dispositivo automático (faster-whisper): cuda
$ .venv/bin/transcriber doctor --require-cuda --backend faster-whisper  →  exit 0
```

`_ctranslate2_cuda_available()` usa `ctranslate2.get_cuda_device_count()`, que consulta
apenas a Driver API e nunca carrega cuBLAS ou cuDNN. A validação pós-upgrade do commit
`8a75bed` passa com a GPU inutilizável — exatamente o caso que ela deveria pegar.

### D3 — Correção de D1 validada

Com `libcublas.so.12` e `libcudnn.so.9` no caminho do carregador, a inferência volta a
funcionar em `sm_120`. O diagnóstico foi feito com uma `libcublas.so.12` que já existia
em outra instalação da máquina, apenas para isolar a causa. **Não é o caminho a
seguir:** a correção publicada instala `nvidia-cublas-cu12` pelo lockfile e pré-carrega
a biblioteca por caminho absoluto. Aproveitar binários CUDA de outro produto instalado
não é redistribuível nem reproduzível.

```
float16        OK  17.50s
int8_float16   OK   1.61s
bfloat16       OK   1.07s
```

Modelo `tiny`, `beam_size=1`, mídia de ~10 s, **GPU em 100 % de uso por outro
processo**. Os tempos provam suporte funcional e nada mais: o `float16` pagou a
inicialização de kernels e a contenção domina o resto. Não usar como linha de base.

O `libcudnn.so.9` já instalado (build cu13) funcionou com o CTranslate2. Não é preciso
`nvidia-cudnn-cu12`, que colidiria com o do PyTorch no mesmo diretório.

### D4 — INT8 e BF16 funcionam na Blackwell

```
ctranslate2.get_supported_compute_types("cuda")
{'int8_float16', 'float16', 'int8_float32', 'int8_bfloat16', 'int8', 'float32', 'bfloat16'}
ctranslate2.get_supported_compute_types("cpu")
{'float32', 'int8', 'int8_float32', 'int16'}
```

Os três tipos testados em D3 executaram de fato. O aviso do README de que "versões
atuais do CTranslate2 desabilitam INT8 nessa arquitetura" está desatualizado.
`compute_type_for()` fixa `float16` e não oferece nenhum dos dois.

### D5 — Nenhuma biblioteca CUDA é encontrável por padrão

```
libcublas.so.12   -> FALHA        libcudnn.so.9  -> FALHA
libcublasLt.so.12 -> FALHA        libcudart.so.12 -> FALHA
```

Com `site-packages/nvidia/cudnn/lib` no `LD_LIBRARY_PATH`, `libcudnn.so.9` passa a
carregar. O PyTorch resolve seus próprios caminhos internamente; o CTranslate2 depende
do carregador do sistema. Logo, a correção precisa de pré-carga explícita por `ctypes`,
e um `dlopen` dessas bibliotecas é uma sonda barata e precisa para o `doctor`.

### D6 — Dependências já estão nas últimas versões

`ctranslate2` 4.8.2 e `faster-whisper` 1.2.1 são as mais recentes no PyPI. Não há
atualização disponível que resolva D1: a solução é conviver com as duas linhas de CUDA.

### D7 — Batch dimensionado para a placa antiga

`fast` usa batch 8 e `quality` batch 4. `_batch_candidates()` apenas divide por 2 após
um OOM e nunca considera subir, então a placa maior não é aproveitada automaticamente
em lugar nenhum — ela só deixa de doer.

### D8 — Benchmarks não são comparáveis entre gerações

O relatório grava apenas `platform.platform()`. Sem nome da GPU, driver, VRAM e
precisão efetiva, a linha de base recomendada no README após o upgrade não pode ser
comparada com nenhuma anterior. Também não registra contenção, então um benchmark
rodado junto com outra carga parece regressão.

### D9 — O recuo de batch não cobre o OOM de carregamento do modelo

Ao validar a Fase 0 com ~2 GiB livres (outro processo ocupava 12 GiB), o perfil `fast`
(modelo `turbo`, FP16) falhou assim:

```
Backend: faster-whisper | modelo: turbo | dispositivo: cuda | precisão: float16 | batch inicial: 8
ERRO: Falha no faster-whisper: CUDA failed with error out of memory
```

O `_batch_candidates()` nunca entrou em ação: o OOM ocorreu dentro de
`WhisperModel(...)`, antes do laço de inferência. Com `--model tiny` na mesma GPU
ocupada, a transcrição completou em 2,92 s com batch 8 e publicou os seis artefatos.

Confirma duas coisas: a correção da Fase 0 funciona (o erro mudou de "biblioteca não
encontrada" para disputa de memória) e o dimensionamento por VRAM livre da Fase 1.2 é
necessário, não cosmético. A mensagem de erro também precisa dizer que outro processo
detinha a placa.

### D10 — O CTranslate2 não substitui precisão silenciosamente

Suspeita levantada e descartada por medição. Pedir uma precisão indisponível levanta
`ValueError`, em qualquer direção:

```
cuda pedido=float16        efetivo=float16
cuda pedido=int8_float16   efetivo=int8_float16
cuda pedido=int16          ValueError: Requested int16 compute type, but the target device...
cpu  pedido=float16        ValueError: Requested float16 compute type, but the target device...
```

Não é preciso validar precisão antes de carregar o modelo. O que faltava era a
mensagem: ela não dizia quais precisões a placa aceita. Agora diz.

### D11 — Os perfis `quality` e `lyrics` falhavam em qualquer mídia acima de 30 s

Bug pré-existente, descoberto ao montar a campanha de medição. O
`BatchedInferencePipeline` do faster-whisper 1.2.1 obtém as fronteiras de segmento do
VAD. Os dois perfis desligam o VAD de propósito (para não descartar áudio cantado ou
baixo) e usam batch 4, combinação que o backend recusa:

```
ERRO: Falha no faster-whisper: No clip timestamps found.
      Set 'vad_filter' to True or provide 'clip_timestamps'.
```

Em `transcribe.py:396-418` o pipeline aceita áudio sem `clip_timestamps` **apenas**
quando a duração é menor que uma janela de 30 s — por isso o bug passava desapercebido
em testes curtos e derrubava qualquer mídia real. Os testes do projeto mockam o
pipeline, então nunca o exerceram. O próprio exemplo do README
(`benchmark video.mp4 --profiles fast quality`) falharia.

Armadilha na correção: quando `clip_timestamps` são fornecidos, cada clipe é
transcrito **só nos primeiros 30 s** (aviso no log, nada mais). Passar um clipe único
cobrindo o arquivo inteiro descartaria quase todo o áudio silenciosamente. A cobertura
precisa ser por janelas de no máximo 30 s.

Medido no MAVIRA (280 s, `large-v3`, FP16, GPU ociosa):

| Modo | Tempo | ×RT | Palavras | Divergência |
| --- | ---: | ---: | ---: | ---: |
| sequencial, batch 1 | 8,2 s | 34,1 | 381 | referência |
| janelas de 30 s, batch 8 | 3,1 s | 91,6 | 373 | 5,0 % |
| janelas de 30 s, batch 16 | 2,7 s | 103,1 | 373 | 5,0 % |

Pelo CLI completo, o perfil `quality` na mesma mídia caiu de 27,09 s para 7,06 s
(3,8×, batch automático 32).

Decisão: a correção segura — batch 1 com aviso — é o padrão, porque a divergência de
5 % nas bordas das janelas é um custo de acurácia que o usuário deve escolher. O ganho
fica disponível em `--window-batching`.

### D12 — As legendas do perfil padrão vinham com granularidade de janela

Segunda descoberta da campanha, e também pré-existente. O `BatchedInferencePipeline`
usa `without_timestamps=True` por padrão, enquanto o caminho sequencial usa `False`. O
projeto não passava a opção, então todo perfil em lote — incluindo o `fast`, que é o
padrão — produzia um bloco de legenda por *chunk*, com o intervalo inteiro do chunk.

Medido no MAVIRA (280 s, `turbo`, batch 8, VAD ligado):

| `without_timestamps` | Tempo | Segmentos | Duração média |
| --- | ---: | ---: | ---: |
| `True` (era o efetivo) | 1,21 s | 1 | 51,3 s |
| `False` (corrigido) | 1,02 s | 6 | 8,5 s |

Não há custo de tempo — a diferença está dentro do ruído. O `.txt` sempre esteve
correto; `.srt` e `.vtt` é que eram inutilizáveis como legenda. Agora o perfil `fast`
produz 12 blocos alinhados à fala nessa mídia, começando em 00:00:02,540.

### D13 — Sem ground truth, nenhum dos modos de inferência é "o certo"

Comparando os modos com os ajustes reais do perfil `quality` (`large-v3`, FP16,
temperaturas até 1.0, `condition_on_previous_text=True`):

| Modo | Tempo | ×RT | Segmentos | Palavras | Divergência |
| --- | ---: | ---: | ---: | ---: | ---: |
| sequencial, batch 1 | 22,6 s | 12,4 | 73 | 490 | referência |
| janelas, sem timestamps | 2,7 s | 102,9 | 10 | 373 | 25,7 % |
| janelas, com timestamps | 5,8 s | 48,5 | 55 | 347 | 31,0 % |

A divergência alta tentaria a conclusão de que as janelas degradam o texto. Mas a
contagem de 5-gramas repetidos mostra que a referência sequencial tem laço de
alucinação — `'i came to you, i'` aparece 8 vezes, contra 13 ocorrências repetidas no
total do modo em janelas (e essas parecem refrão real). O sequencial produz mais
palavras em parte porque repete.

Conclusão honesta: as 143 palavras de diferença não medem perda de conteúdo, e sem uma
transcrição de referência humana não é possível dizer qual modo é mais fiel. Por isso
o padrão continua sendo o modo seguro (batch 1) e o ganho fica em `--window-batching`,
com esses números documentados. Para decidir de verdade seria preciso uma referência
humana de uma das mídias.

### D14 — Mais VRAM não compra batch maior (e o auto-dimensionamento piorava tudo)

Campanha de 2026-10-10 com a GPU ociosa. Bancada: perfil `quality` (VAD desligado) com
`--window-batching` sobre 1120 s de vocal contínuo (MAVIRA concatenado 4×), duas
repetições por configuração, melhor tempo reportado.

| Configuração | Precisão | Batch | Melhor | Pior | ×RT | VRAM pico | Divergência |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| large-v3 sequencial | FP16 | 1 | 74,11 s | 89,50 s | 15,1 | 5420 MiB | referência |
| large-v3 | FP16 | 4 | **26,78 s** | 27,40 s | 41,8 | 4940 MiB | 23,8 % |
| large-v3 | FP16 | 8 | 26,88 s | 28,03 s | 41,7 | 6028 MiB | 23,8 % |
| large-v3 | BF16 | 8 | 28,13 s | 28,44 s | 39,8 | 6092 MiB | 27,0 % |
| large-v3 | INT8+FP16 | 8 | 30,68 s | 31,14 s | 36,5 | **4364 MiB** | 22,6 % |
| large-v3 | FP16 | 16 | 30,38 s | 30,56 s | 36,9 | 8204 MiB | 23,8 % |
| large-v3 | FP16 | 32 | 54,28 s | 54,53 s | 20,6 | 12588 MiB | 23,8 % |
| turbo sequencial | FP16 | 1 | 28,15 s | 32,38 s | 39,8 | 2380 MiB | referência |
| turbo | FP16 | 4 | **20,79 s** | 20,99 s | 53,9 | 2732 MiB | 13,5 % |
| turbo | BF16 | 8 | 19,38 s | 19,92 s | 57,8 | 3276 MiB | 19,5 % |
| turbo | FP16 | 8 | 23,22 s | 23,25 s | 48,2 | 3276 MiB | 13,5 % |
| turbo | INT8+FP16 | 8 | 24,73 s | 24,86 s | 45,3 | **2316 MiB** | 13,3 % |
| turbo | FP16 | 16 | 27,97 s | 28,08 s | 40,0 | 4332 MiB | 13,4 % |
| turbo | FP16 | 32 | 51,36 s | 52,47 s | 21,8 | 6060 MiB | 13,5 % |

Três conclusões, todas contrariando a intuição de "placa maior, batch maior":

1. **O joelho está em 4–8.** De 8 para 32 o tempo dobra, e o `turbo` em batch 32
   (51,36 s) fica mais lento que em batch 1 (28,15 s). O pico de VRAM confirma que
   caberia — 12 588 MiB de 16 311 — então o limite não é memória. A causa provável é
   pressão de alocação/banda com 32 janelas de 30 s simultâneas.
2. **O auto-dimensionamento da Fase 1.2 estava errado na direção de subida.** Ele
   escolhia 32 com a GPU livre, o que custava 2,0× no `large-v3` e 2,5× no `turbo`
   contra o valor do perfil. Regressão introduzida por mim e corrigida: o planejamento
   agora só reduz, nunca sobe, e `MAXIMUM_AUTOMATIC_BATCH` caiu de 32 para 8.
3. **O ganho real do lote é 2,8×** no `large-v3` (74,11 s → 26,78 s) e 1,35× no
   `turbo`. É isso que a placa nova entrega: rodar `large-v3` em lote, não um batch
   grande.

Precisão: FP16 é a mais rápida no `large-v3`; no `turbo` o BF16 foi 17 % mais rápido
(19,38 s contra 23,22 s, repetições apertadas), mas divergiu mais do sequencial
(19,5 % contra 13,5 %) — sem ground truth não há como saber se é melhor ou pior, então
o padrão continua FP16. INT8+FP16 é a opção de memória, não de velocidade: ~28 % menos
VRAM por ~14 % mais tempo.

As estimativas de `vram.py` se mostraram bem calibradas: para `large-v3` FP16 previ
5770/7850/12010 MiB em batch 8/16/32 e medi 6028/8204/12588.

### D15 — `htdemucs_ft` custa tempo, não VRAM

Fase 3.1, medida em 2026-10-10 com a GPU ociosa. Mídia: MAVIRA (280 s), perfil
`lyrics` completo (separação, transcrição, revisão e reavaliação).

| Separação | Tempo | ×RT | VRAM pico | Segmentos | Suspeitos | Reavaliados |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `htdemucs` | 27,3 s | 10,2 | 4140 MiB | 57 | 19 | 2 |
| `htdemucs_ft` | 66,7 s | 4,2 | 4172 MiB | 57 | 11 | 2 |

Divergência de texto entre as duas: 1,50 % (379 palavras em ambas).

A premissa registrada no TODO original — de que o ensemble era "caro demais" para a
placa antiga — estava errada. Ele não consome VRAM adicional, porque os quatro modelos
rodam em sequência; o custo é só de tempo.

**Conclusão inicial retratada.** Esta medição tinha n=1 por modelo, e a partir dela eu
afirmei que o `htdemucs_ft` reduzia a fila de revisão em 42 % (19 → 11) e troquei o
padrão do perfil `lyrics`. Repetindo com n=3 por modelo na mesma mídia:

| Separação | Suspeitos por execução | Média | Segmentos | Palavras |
| --- | --- | ---: | --- | --- |
| `htdemucs` | 16, 22, 22 (+19 da campanha) | 19,8 | 60, 60, 60 | 379 em todas |
| `htdemucs_ft` | 19, 17, 19 (+11 da campanha) | 17,0 | 57, 55, 57 | 379, 380, 379 |

As faixas se sobrepõem (16–22 contra 11–19): o `11` era o extremo inferior do próprio
ruído do `htdemucs_ft`, e o `19` o extremo superior do `htdemucs`. A diferença de
médias não sustenta pagar 2,4× no tempo. **O padrão do perfil `lyrics` voltou para
`htdemucs`**; `--demucs-model htdemucs_ft` continua disponível.

### D17 — A contagem da fila de revisão é métrica ruidosa

Derivado de D15, e vale para qualquer medição futura. Nas seis execuções acima o
**texto foi idêntico** (379 palavras em cinco das seis), mas a contagem de segmentos
suspeitos variou de 16 a 22 no mesmo modelo — ±27 %.

A causa é estrutural: `quality_reasons()` decide por comparação com limiar
(`avg_logprob < -1.0`, `compression_ratio > 2.4`, `no_speech_prob > 0.6`). Segmentos
cujos valores ficam perto do limiar trocam de lado entre execuções por variação
numérica da GPU, sem que o texto mude.

Consequência prática: nunca compare duas configurações por uma única execução dessa
contagem. Para uma diferença de ~15 % ser detectável, são necessárias várias repetições
por braço — e foi exatamente esse erro que produziu a conclusão retratada em D15.

### D16 — O VAD descartava a mídia em silêncio (perfil padrão)

Relatado pelo usuário em 2026-10-10: os `.srt` de dois vídeos saíram "faltando
trechos". Ambos foram gerados com o perfil `fast` (padrão, VAD ligado, `turbo`).

| Mídia | Duração | Áudio mantido pelo VAD | Segmentos | Palavras | Tempo |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAVIRA | 280 s | 11,9 s (4,3 %) | 6 | 23 | 4,4 s |
| Hypnotic Vocal Flow | 6941 s | 69,8 s (1,0 %) | 32 | 161 | 36,2 s |

Não houve erro: o programa reportou sucesso tendo transcrito 1 % de um vídeo de quase
duas horas. O dado estava disponível (`duration_after_vad`, já presente no `.json` de
saída) e não era usado para nada.

Dois sintomas, uma causa. Além da perda de texto, os segmentos sobreviventes ficam com
duração distorcida: no MAVIRA, o bloco 3 cobre 8,32 s–44,26 s com três palavras, porque
os vãos descartados são absorvidos no fim de cada legenda. Os seis segmentos
compartilham o mesmo `avg_logprob` (−0,261), sinal de que vieram de uma decodificação
única sobre os trechos que o VAD aceitou.

Comparação na mesma mídia (MAVIRA, GPU ociosa):

| Perfil | Segmentos | Palavras | Último fim | 5-gramas repetidos 3+× |
| --- | ---: | ---: | ---: | ---: |
| `fast` (VAD ligado) | 6 | 23 | 53,8 s | — |
| `quality` (sem VAD) | 61 | 445 | 280,0 s | 52 |
| `lyrics` (Demucs, sem VAD) | 57 | 379 | 268,7 s | 16 |

O `quality` recupera mais palavras, mas 52 de seus 5-gramas repetem 3 vezes ou mais —
inclui o laço `'i came to you, i'` 8×, já visto em D13. O `lyrics` é o perfil correto
para música.

Correção: `vad_coverage_notice()` avisa quando o VAD mantém menos de 50 % do áudio,
nomeando os números e a alternativa, e o `metadata.json` passa a registrar
`audio_seconds` e `audio_after_vad_seconds` para auditar execuções passadas.

## Verificado e descartado

- `whisper._MODELS["large"] == _MODELS["large-v3"]`: o perfil `legacy` já usa o
  large-v3. Não há ganho em trocar o nome do modelo.
- Mídias deste diretório como carga de medição com VAD: todas são musicais e o
  Silero detecta de 1,8 % a 7,7 % de fala (o clipe de 20 min de `Hypnotic Vocal Flow`
  rendeu 20,1 s de fala e 48 palavras). Qualquer ×RT medido com VAD ligado nessas
  mídias mede o VAD, não a transcrição. A campanha usa VAD desligado.
- Passar um `clip_timestamps` único cobrindo o arquivo: descartaria tudo além dos
  primeiros 30 s (D11).
- `torch.cuda.mem_get_info()` como fonte de VRAM livre: preterido em favor de
  `nvidia-smi`, que não cria contexto CUDA no processo apenas para responder e já
  reporta a memória retida por outros processos — que é exatamente o que importa nesta
  máquina.
- Validação prévia de `--compute-type` contra a lista suportada: desnecessária, o
  CTranslate2 já falha com `ValueError` (D10).
- PyTorch: `torch.cuda.get_arch_list()` inclui `sm_120`. O caminho torch
  (openai-whisper e Demucs) está correto para esta placa; o problema é só do
  CTranslate2.

## Comandos de reprodução

```bash
# Ambiente, precisões suportadas e VRAM
.venv/bin/python -c "import ctranslate2, torch; \
print(ctranslate2.get_supported_compute_types('cuda')); \
print(torch.cuda.get_device_capability(0), torch.cuda.get_arch_list()); \
print([x/2**30 for x in torch.cuda.mem_get_info(0)])"

# Sonda das bibliotecas (D5)
.venv/bin/python -c "
import ctypes
for lib in ('libcublas.so.12','libcublasLt.so.12','libcudnn.so.9'):
    try: ctypes.CDLL(lib); print(lib,'ok')
    except OSError: print(lib,'FALHA')"

# Reprodução de D1/D3 (requer modelo tiny em cache)
.venv/bin/python -c "
from faster_whisper import WhisperModel
m = WhisperModel('tiny', device='cuda', compute_type='float16', local_files_only=True)
print(' '.join(s.text for s in m.transcribe('video.mp4', language='en')[0]))"
```

## Linha de base de desempenho

Estabelecida em 2026-10-10 com a GPU ociosa (nenhum processo de compute concorrente,
14,3 GiB livres). Mídia: MAVIRA concatenado 4× (1120 s de vocal contínuo), perfil
`quality` com `--window-batching`, duas repetições, melhor tempo. A tabela completa
está em D14.

| Data | Modelo | Precisão | Batch | Tempo | × tempo real | VRAM pico |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2026-10-10 | large-v3 | FP16 | 4 | 26,78 s | 41,8 | 4940 MiB |
| 2026-10-10 | large-v3 | FP16 | 1 (sequencial) | 74,11 s | 15,1 | 5420 MiB |
| 2026-10-10 | turbo | FP16 | 4 | 20,79 s | 53,9 | 2732 MiB |
| 2026-10-10 | turbo | FP16 | 1 (sequencial) | 28,15 s | 39,8 | 2380 MiB |

Ainda falta WER: não há transcrição humana de referência (Fase 5 do TODO). As colunas
de divergência medem diferença entre duas saídas do próprio modelo, não acurácia.

## Histórico de progresso

| Data | Fase | O que mudou |
| --- | --- | --- |
| 2026-10-10 | — | Auditoria de hardware pós-upgrade; D1–D8 registradas; `TODO.md`, `AGENTS.md` e este arquivo criados. |
| 2026-10-10 | 3.1 | `htdemucs_ft` medido (D15). Primeira conclusão (fila 42 % menor) **retratada** ao repetir com n=3: a diferença fica dentro do ruído (D17). Padrão do `lyrics` revertido para `htdemucs`. |
| 2026-10-10 | 0.5 | Perda silenciosa de áudio pelo VAD (D16), relatada pelo usuário: aviso de cobertura implementado e cobertura registrada no `metadata.json`. |
| 2026-10-10 | 1.1, 1.3 | Campanha com GPU ociosa (D14): joelho de batch em 4–8, `MAXIMUM_AUTOMATIC_BATCH` de 32 para 8, subida de batch removida (era regressão de até 2,5×), FP16 mantido como padrão. Linha de base estabelecida. |
| 2026-10-10 | 0.3, 0.4 | Bugs pré-existentes D11 (lote sem VAD falhava acima de 30 s) e D12 (legendas com granularidade de janela) corrigidos; `--window-batching` adicionado. |
| 2026-10-10 | 1.2 | Batch dimensionado pela VRAM livre (`vram.py`), OOM de carregamento com mensagem acionável, VRAM e batch planejado no `metadata.json`. Validado na GPU disputada: `turbo` FP16 recusado com diagnóstico completo e `turbo` int8_float16 concluído em 6,09 s com batch 8→1. |
| 2026-10-10 | 2 | `doctor` passa a reportar compute capability, arquiteturas do PyTorch, precisões suportadas e VRAM livre; relatório de benchmark ganhou bloco `hardware`, VRAM inicial e contenção, sem caminhos pessoais. |
| 2026-10-10 | 1.1 (parcial) | Erro de precisão não suportada agora lista as opções reais da placa (D10). Default FP16 mantido até haver medição com GPU ociosa. |
| 2026-10-10 | 4 | README corrigido: aviso de INT8 na Blackwell removido, convivência cu12/cu13 documentada, `--smoke-test` e dimensionamento por VRAM descritos. |
| 2026-10-10 | 0 | Fase 0 concluída: `cuda_runtime.py` pré-carrega cuBLAS 12 e cuDNN 9; sonda do `doctor` separa driver visível de bibliotecas carregáveis; `doctor --smoke-test` adicionado; `nvidia-cublas-cu12==12.9.2.10` e `nvidia-cuda-nvrtc-cu12==12.9.86` fixados no lock. GPU validada de ponta a ponta. D9 registrada. 55 testes passando. |
