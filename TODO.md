# TODO

Plano de melhorias derivado da auditoria de hardware de 2026-10-10, feita depois da
troca para a RTX 5060 Ti (Blackwell, `sm_120`, 16 GiB). O diagnóstico completo, com as
evidências medidas, está em [reference/Progress_Control.md](reference/Progress_Control.md).

As fases estão em ordem de dependência: a Fase 0 destrava o caminho GPU, e sem ela
nenhum ajuste de batch ou precisão tem efeito observável.

## Fase 0 — Destravar a GPU (bloqueante) — CONCLUÍDA em 2026-10-10

O backend `faster-whisper` não executa em GPU neste ambiente: o CTranslate2 4.8.2 é
compilado contra CUDA 12 e pede `libcublas.so.12`, que não existe no sistema nem no
virtualenv — o `requirements.lock` instala apenas a linha cu13. Os perfis `fast`,
`quality` e `lyrics` falham logo após carregar o modelo.

Validada na GPU real: `doctor --require-cuda --smoke-test` retorna 0 e uma
transcrição completa publica artefatos em CUDA, sem `LD_LIBRARY_PATH`. Ver D3 e D9
no controle de progresso.

### 0.1 Resolver as bibliotecas CUDA do CTranslate2

- [x] Adicionar `nvidia-cublas-cu12` ao `requirements.lock` e às dependências do
      `pyproject.toml`. Ele convive com a linha cu13 do PyTorch: os *sonames* são
      distintos (`libcublas.so.12` e `libcublas.so.13`).
- [x] Criar `src/transcriber/cuda_runtime.py` com uma função idempotente que faz
      `ctypes.CDLL` de `libcublasLt.so.12`, `libcublas.so.12` e `libcudnn*.so.9`
      a partir dos diretórios dentro de `site-packages/nvidia/`, **antes** do primeiro
      `import ctranslate2`.
- [x] Chamar essa função no ponto de entrada do backend
      ([`backends/faster_whisper.py`](src/transcriber/backends/faster_whisper.py)) e na
      sonda de hardware, nunca no `__init__.py` do pacote — o CLI em CPU e os testes
      não devem pagar por carregamento de CUDA.
- [x] Não depender de `LD_LIBRARY_PATH`: a correção precisa funcionar com o comando
      publicado (`.venv/bin/transcriber`), sem variáveis de ambiente.

**Aceite:** `WhisperModel("tiny", device="cuda", compute_type="float16")` transcreve sem
`LD_LIBRARY_PATH` manual, e o `cpu` continua funcionando em máquina sem NVIDIA.

Reutilizar o cuDNN 9 já instalado (build cu13) — foi validado com o CTranslate2 e
evita `nvidia-cudnn-cu12`, que colidiria com o do PyTorch no mesmo diretório.

### 0.2 Tornar o `doctor` capaz de detectar essa falha

Hoje `doctor --require-cuda --backend faster-whisper` retorna 0 com a GPU inutilizável:
a sonda usa `ctranslate2.get_cuda_device_count()`, que só consulta a Driver API e nunca
toca em cuBLAS ou cuDNN.

- [x] Em [`hardware.py`](src/transcriber/hardware.py), separar dois níveis de sonda:
      *driver visível* (atual) e *bibliotecas carregáveis* (`dlopen` de cuBLAS 12,
      cuBLASLt 12 e cuDNN 9), reportando os dois no relatório.
- [x] Adicionar um teste de fumaça opcional (`--smoke-test`) que carrega o modelo
      `tiny` em CUDA e transcreve ~1 s de áudio sintético. É o único jeito honesto de
      afirmar que a GPU funciona.
- [x] `--require-cuda` deve falhar (código 1) quando o driver aparece mas as
      bibliotecas não carregam, com mensagem apontando o pacote que falta.

**Aceite:** com `nvidia-cublas-cu12` removido do ambiente, `doctor --require-cuda`
retorna 1 e nomeia `libcublas.so.12` como ausente.

## Fase 0.3 — Inferência em lote sem VAD — CONCLUÍDA em 2026-10-10

Bug pré-existente descoberto durante a campanha de medição (D11): os perfis `quality` e
`lyrics` desligam o VAD e usam batch 4, e o `BatchedInferencePipeline` recusa áudio
acima de 30 s sem fronteiras de segmento. Qualquer mídia real falhava.

- [x] Recuo seguro para batch 1 quando o VAD está desligado, com mensagem explicando
      o motivo e apontando a alternativa.
- [x] Recuperar também a recusa vinda do backend, caso ela surja por outro caminho.
- [x] `--window-batching`: cobre o áudio com janelas de no máximo 30 s e mantém o
      lote. Opcional e desligado por padrão, porque custa ~5 % de divergência de texto
      nas bordas.
- [x] Testes cobrindo as janelas, o recuo e a recusa do backend — os testes antigos
      mockavam o pipeline e por isso nunca exerceram esse caminho.

**Medido:** perfil `quality` no MAVIRA caiu de 27,09 s para 7,06 s (3,8×) com
`--window-batching` e batch automático 32.

## Fase 0.4 — Granularidade das legendas em lote — CONCLUÍDA em 2026-10-10

Bug pré-existente (D12): o pipeline em lote descarta os tokens de timestamp por
padrão, então `.srt` e `.vtt` saíam com um bloco por *chunk*. Afetava o perfil `fast`,
que é o padrão.

- [x] Passar `without_timestamps=False` explicitamente nos dois caminhos de inferência.
- [x] Teste garantindo a opção no caminho em lote e no sequencial.

**Medido:** sem custo de tempo (1,02 s contra 1,21 s, dentro do ruído) e de 1 segmento
de 51,3 s para 6 de 8,5 s em média.

## Fase 1 — Aproveitar a arquitetura e os 16 GiB

### 1.1 Precisões novas (`bfloat16`, `int8_float16`) — CONCLUÍDA em 2026-10-10

[`config.py:98`](src/transcriber/config.py#L98) fixa `float16` em qualquer GPU. Nesta
placa o CTranslate2 reporta e executa também `bfloat16` e `int8_float16` — o aviso do
README sobre INT8 indisponível na Blackwell está desatualizado.

- [x] ~~Fazer `compute_type_for()` escolher conforme a arquitetura~~ — **descartado
      por medição.** FP16 é a mais rápida no `large-v3`; o BF16 foi 17 % mais rápido no
      `turbo`, mas divergiu mais do sequencial, e sem ground truth isso não autoriza
      trocar o padrão. INT8+FP16 economiza ~28 % de VRAM por ~14 % mais tempo: é opção
      de memória, não de velocidade. `doctor` lista as precisões disponíveis e
      `--compute-type` continua soberano.
- [x] Nunca silenciar um `--compute-type` explícito não suportado: erro claro, não
      substituição automática.
- [x] Medir `float16` × `bfloat16` × `int8_float16` com a GPU **ociosa** antes de mudar
      qualquer default de perfil. Feito em 2026-10-10 (D14).

**Aceite:** benchmark dos três tipos na mesma mídia, com WER, registrado em
`reference/Progress_Control.md`; default alterado somente se houver ganho medido.

**Bloqueado por GPU ocupada.** O erro de precisão não suportada já lista as opções
reais da placa, e `doctor` as reporta. O default continua FP16 de propósito: mudá-lo
sem medição seria adivinhação. Retomar quando a placa estiver ociosa.

### 1.2 Batch guiado pela VRAM livre — CONCLUÍDA em 2026-10-10

[`faster_whisper.py:57`](src/transcriber/backends/faster_whisper.py#L57) só sabe
descer: divide o batch por 2 após um OOM e nunca considera subir. Com 16 GiB, `fast`
em batch 8 e `quality` em batch 4 são herança da placa antiga.

- [x] Consultar `torch.cuda.mem_get_info()` antes de carregar o modelo e derivar o
      batch inicial da memória **livre**, não da total — a GPU é compartilhada com
      outras cargas de IA.
- [x] Tratar o OOM no **carregamento do modelo**, que acontece antes do laço de recuo
      de batch e por isso não é recuperável hoje (descoberta D9). A mensagem precisa
      dizer quanta VRAM estava livre e quanto o modelo exigia, em vez de apenas
      `CUDA failed with error out of memory`.
- [x] Manter o recuo por OOM como rede de segurança, não como estratégia principal.
- [x] Registrar no `metadata.json` a VRAM livre observada e o batch escolhido, para
      que uma execução lenta possa ser explicada depois.

**Aceite:** com a GPU livre o batch sobe acima do valor do perfil; com a GPU ocupada
(~2 GiB livres) o job escolhe um batch que caiba em vez de estourar no meio.

Implementado em [`vram.py`](src/transcriber/vram.py). A leitura usa `nvidia-smi` em vez
de `torch.cuda.mem_get_info()`: a consulta precisa incluir a memória retida por *outros*
processos e não deve criar um contexto CUDA só para responder.

**Revisado em 2026-10-10 pela campanha.** A redução conforme a VRAM livre se confirmou
útil e está validada na GPU real. A *subida* de batch, que esta fase também introduziu,
foi removida: ela custava até 2,5× de desempenho com a GPU livre (D14). Abundância de
VRAM não é motivo para aumentar o batch.

### 1.3 Perfis revisados para 16 GiB — CONCLUÍDA em 2026-10-10

- [x] Reavaliar os batches. Resultado oposto ao esperado: o joelho está em 4–8 e o
      batch 32 dobra o tempo, apesar de caber na VRAM (D14). Os valores dos perfis
      (`fast` 8, `quality` e `lyrics` 4) já estavam na faixa boa; o que mudou foi o
      teto automático, de 32 para 8, e a remoção da subida de batch — que era uma
      regressão de até 2,5× introduzida na 1.2.
- [ ] Verificar se o perfil `fast` ganha ao baixar de 8 para 4. Na bancada sem VAD, o
      `turbo` em batch 4 foi 10 % mais rápido que em 8 (20,79 s contra 23,22 s), mas o
      `fast` roda com VAD ligado, caminho em que não há medição confiável — as mídias
      do projeto rendem de 1,8 % a 7,7 % de fala detectada. Para medir, construa uma
      mídia concatenando apenas os trechos que o VAD aceita e repita a varredura.

## Fase 2 — Observabilidade do hardware — CONCLUÍDA em 2026-10-10

### 2.1 `doctor` mais completo

Faltam exatamente os campos que importam numa troca de placa.

- [x] Compute capability (`sm_120`) e lista de arquiteturas do PyTorch.
- [x] Precisões suportadas por CPU e GPU segundo o CTranslate2.
- [x] VRAM livre e total, além dos processos que ocupam a GPU.
- [x] Divergência entre a CUDA do PyTorch e a exigida pelo CTranslate2, em destaque —
      é a causa da falha da Fase 0.

### 2.2 Benchmarks comparáveis entre gerações

[`benchmark.py:160`](src/transcriber/benchmark.py#L160) grava apenas
`platform.platform()`. Sem identificar a GPU, a linha de base que o README manda
registrar após o upgrade não pode ser comparada com nada anterior.

- [x] Bloco `hardware` no relatório: nome da GPU, driver, CUDA, VRAM total, compute
      capability, versões de PyTorch/CTranslate2 e precisão efetiva.
- [x] Registrar contenção: VRAM livre no início e se havia outros processos de compute
      na GPU. Um benchmark rodado junto com outra carga parece regressão sem isso.
- [x] Preservar a ausência de caminhos pessoais, como já faz o `doctor --json`.

**Aceite:** dois relatórios de GPUs diferentes podem ser comparados apenas pelo JSON.

## Fase 3 — Qualidade habilitada pela VRAM nova

### 3.1 `htdemucs_ft` no perfil `lyrics` — CONCLUÍDA em 2026-10-10

- [x] Validar tempo e pico de VRAM. A premissa de que era "caro demais" estava errada:
      não consome VRAM adicional (4172 MiB contra 4140 MiB), só tempo (66,7 s contra
      27,3 s). Os quatro modelos rodam em sequência.
- [x] Decidido: virou o padrão do perfil `lyrics`. A fila de revisão caiu de 19 para 11
      segmentos suspeitos (−42 %) com 1,5 % de divergência de texto, e a revisão manual
      é o trabalho real desse perfil. `--demucs-model htdemucs` prioriza velocidade.

### 3.2 Sobrepor separação e transcrição em lote

[`separation.py:70`](src/transcriber/separation.py#L70) descarrega o Demucs antes de
carregar o Whisper — correto em 8 GiB, desnecessário em 16.

- [ ] Em lote, separar o arquivo N+1 enquanto o N é transcrito.
- [ ] Só avançar nisso depois da Fase 1: a sobreposição exige saber quanta VRAM cada
      etapa realmente usa, e só vale a pena se a GPU estiver ociosa. Com os números de
      D15 isso já é dimensionável: a separação usa ~4,2 GiB e a transcrição em lote 4
      usa ~4,9 GiB, então as duas etapas cabem simultaneamente em 16 GiB.

## Fase 5 — Referência humana para decidir acurácia

A campanha mostrou que não há como comparar modos de inferência sem ground truth: o
modo sequencial alucina repetições e o modo em janelas produz menos palavras, e os dois
efeitos se confundem na mesma métrica (D13).

- [ ] Transcrever manualmente uma das mídias (ou obter a letra oficial) e salvá-la como
      referência, para que `benchmark --reference` passe a medir WER de verdade em vez
      de divergência entre duas saídas possivelmente erradas.
- [ ] Com essa referência, decidir se `--window-batching` deve ser o padrão do perfil
      `quality` e se o `lyrics` deve manter `condition_on_previous_text=False`.

## Fase 4 — Documentação — CONCLUÍDA em 2026-10-10

- [x] Corrigir o aviso sobre INT8 na Blackwell no [README](README.md) (seção
      "Perfis").
- [x] Trocar "instale CUDA 12/cuDNN 9 no sistema" pela convivência real entre a linha
      cu13 do PyTorch e a cu12 do CTranslate2.
- [x] Atualizar a seção de validação pós-upgrade com o `--smoke-test` da 0.2.
