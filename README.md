# Buddies x International Students Matching

Sistema de matching otimizado para o Programa Buddy, que conecta estudantes brasileiros (Buddies) com estudantes internacionais de intercâmbio.

## 📁 Estrutura do Projeto

```
buddies-matching/
├── match_improved.py      # Script principal de matching
├── match.py                # Script legado (versão antiga)
├── utils.py                 # Funções utilitárias de processamento
├── debug.py                # Inspeciona colunas/abas de um .xlsx já separado
├── classes/
│   ├── optimizer.py        # Otimizador legado
│   └── improved_optimizer.py  # Otimizador melhorado
├── models/
│   ├── config.py           # Configurações do matching
│   └── person.py           # Classe Person
├── scripts/                 # Scripts auxiliares
│   ├── explore_data.py     # Explora a estrutura do arquivo bruto (respostas do form)
│   ├── split_by_gender.py  # Separar dados por gênero (flexível, com --name-column etc.)
│   ├── run_split.py        # Script rápido de separação (descobre o .xlsx automaticamente)
│   └── analyze.py          # Análise da proporção buddy/internacional após o split
└── files/
    ├── data/                # Dados de entrada (Excel)
    └── results/             # Resultados do matching
```

## 🚀 Passo a Passo Completo

### 0. Organizar os arquivos de entrada

Coloque o `.xlsx` exportado do formulário em uma pasta, por exemplo `files/data/26.1/`, e crie um arquivo `women_names.txt` na mesma pasta com um nome feminino por linha (todos os outros nomes serão tratados como masculinos):

```
Maria Santos
Ana Paula
Alaya Minet
```

### 1. (Opcional) Explorar os dados brutos

Para inspecionar as colunas, valores únicos e comentários do arquivo exportado antes de processá-lo:

```bash
python scripts/explore_data.py "./files/data/26.1" --pattern "*.xlsx" --output "./files/data/26.1/data_structure.txt"
```

| Argumento | Descrição | Padrão |
|-----------|-----------|--------|
| `data_dir` | Pasta onde está o `.xlsx` (posicional, obrigatório) | - |
| `--pattern` | Padrão glob para localizar o arquivo | `*2026*.xlsx` |
| `-o`, `--output` | Arquivo de saída com o resumo da estrutura | `data_structure.txt` |

### 2. Separar os dados por gênero

Use o script rápido `run_split.py`, que descobre automaticamente o `.xlsx` e o `women_names.txt` dentro da pasta informada:

```bash
python scripts/run_split.py "./files/data/26.1"
```

Por padrão, o `data_split.xlsx` é salvo na mesma pasta de entrada. Para salvar em outra pasta, use `-o`/`--output-folder`:

```bash
python scripts/run_split.py "./files/data/26.1" --output-folder "./files/results/26.1"
```

| Argumento | Descrição | Padrão |
|-----------|-----------|--------|
| `folder` | Pasta com o `.xlsx` de entrada e o `women_names.txt` (posicional, obrigatório) | - |
| `-o`, `--output-folder` | Pasta onde salvar `data_split.xlsx` | mesma pasta de entrada |

Alternativamente, use `split_by_gender.py` para mais controle sobre os caminhos e a coluna de nomes:

```bash
python scripts/split_by_gender.py --input "./files/data/26.1/dados.xlsx" --women-file "./files/data/26.1/women_names.txt" --output "./files/data/26.1/data_split.xlsx"
```

| Argumento | Descrição | Padrão |
|-----------|-----------|--------|
| `--input`, `-i` | Caminho do arquivo Excel de entrada (obrigatório) | - |
| `--women-file`, `-w` | Caminho do arquivo com nomes femininos (obrigatório) | - |
| `--output`, `-o` | Caminho do arquivo de saída | `<input>_split.xlsx` |
| `--name-column` | Coluna com os nomes dos participantes | `"Tell us what's you name: "` |
| `--sheet` | Aba específica a ler do arquivo de entrada | primeira aba |

### 3. (Opcional) Verificar o resultado do split

Confira rapidamente as colunas e a proporção buddy/internacional do arquivo gerado:

```bash
python debug.py "./files/data/26.1/data_split.xlsx" --sheet Men
python scripts/analyze.py "./files/data/26.1/data_split.xlsx"
```

| Script | Argumento | Descrição | Padrão |
|--------|-----------|-----------|--------|
| `debug.py` | `xlsx_path` | Caminho do `.xlsx` separado (posicional, obrigatório) | - |
| `debug.py` | `--sheet` | Aba a inspecionar | `Men` |
| `scripts/analyze.py` | `xlsx_path` | Caminho do `.xlsx` separado (posicional, obrigatório) | - |

### 4. Executar o Matching

Para mulheres:
```bash
python match_improved.py --xlsx-path "./files/data/26.1/data_split.xlsx" --sheet "Women" --s m --to-excel --save-path "./files/results/26.1"
```

Para homens:
```bash
python match_improved.py --xlsx-path "./files/data/26.1/data_split.xlsx" --sheet "Men" --s h --to-excel --save-path "./files/results/26.1"
```

### 5. Argumentos Disponíveis (`match_improved.py`)

| Argumento | Descrição | Padrão |
|-----------|-----------|--------|
| `--xlsx-path` | Caminho do arquivo Excel | `data/dados.xlsx` |
| `--sheet` | Nome da aba no Excel | `Sheet1` |
| `--s` | Gênero (`m`=mulher, `h`=homem) | obrigatório |
| `--to-excel` | Salvar resultados em Excel | - |
| `--to-csv` | Salvar resultados em CSV | - |
| `--save-path` | Pasta para salvar resultados | `.` |
| `--comment-weight` | Peso para similaridade de comentários | `0.1` |
| `--buddy-weight` | Peso para similaridade buddy-estudante | `0.7` |
| `--student-weight` | Peso para similaridade estudante-estudante | `0.3` |
| `--comfort-bonus` | Bônus para conforto com diferenças | `0.1` |
| `--comfort-penalty` | Penalidade para desconforto | `0.1` |
| `--legacy` | Usar algoritmo antigo | - |

## ✨ Funcionalidades

### Matching Melhorado (`match_improved.py`)

1. **Similaridade de Comentários**: Usa TF-IDF para encontrar participantes com comentários similares
2. **Modificador de Conforto**: Considera se a pessoa está confortável com diferenças culturais
3. **Análise de Coesão de Grupo**: Avalia a similaridade entre estudantes internacionais no mesmo grupo
4. **Configuração Flexível**: Todos os pesos são ajustáveis via linha de comando

### Algoritmo de Otimização

O sistema usa **Programação Linear Inteira (PuLP)** para maximizar:
- Similaridade cosseno entre buddy e estudante
- Similaridade entre estudantes do mesmo grupo

Restrições:
- Cada estudante internacional recebe exatamente 1 buddy
- Cada buddy recebe entre `min` e `max` estudantes (calculado pela razão)

## 📊 Arquivos de Saída

Após o matching:
- `resultados_{gênero}.xlsx` - Atribuições finais
- `group_summary_{gênero}.xlsx` - Resumo dos grupos com coesão
- `cos_similarity/cosine_similarity_{gênero}.xlsx` - Matriz de similaridade base
- `cos_similarity/final_similarity_{gênero}.xlsx` - Matriz com bônus aplicados

## 🔧 Dependências

```bash
pip install pandas scikit-learn pulp openpyxl
```

## 📝 Exemplo Completo

```bash
# 0. Criar arquivo com nomes das mulheres dentro da pasta de dados
echo "Maria Santos
Ana Paula
..." > "./files/data/26.1/women_names.txt"

# 1. (Opcional) Explorar a estrutura do arquivo bruto
python scripts/explore_data.py "./files/data/26.1" --pattern "*.xlsx"

# 2. Separar dados por gênero
python scripts/run_split.py "./files/data/26.1" --output-folder "./files/data/26.1"

# 3. (Opcional) Verificar o resultado do split
python debug.py "./files/data/26.1/data_split.xlsx"
python scripts/analyze.py "./files/data/26.1/data_split.xlsx"

# 4. Executar matching para mulheres
python match_improved.py --xlsx-path "./files/data/26.1/data_split.xlsx" --sheet "Women" --s m --to-excel --save-path "./files/results/26.1"

# 5. Executar matching para homens
python match_improved.py --xlsx-path "./files/data/26.1/data_split.xlsx" --sheet "Men" --s h --to-excel --save-path "./files/results/26.1"
```

## 📌 Notas Importantes

- A coluna de nomes deve ser `"Tell us what's you name: "` (ou será detectada automaticamente)
- A coluna de tipo deve conter `"Brazilian student (Buddy)"` ou `"International student (Incoming)"`
- Comentários triviais (`.`, `/`, emojis) são ignorados no cálculo de similaridade
- Todos os scripts recebem caminhos de arquivo/pasta via linha de comando — nenhum caminho fica fixo no código, então o projeto funciona em qualquer máquina ou pasta de dados
