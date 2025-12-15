# Buddies x International Students Matching

Sistema de matching otimizado para o Programa Buddy, que conecta estudantes brasileiros (Buddies) com estudantes internacionais de intercâmbio.

## 📁 Estrutura do Projeto

```
buddies-matching/
├── match_improved.py      # Script principal de matching
├── match.py               # Script legado (versão antiga)
├── utils.py               # Funções utilitárias de processamento
├── classes/
│   ├── optimizer.py       # Otimizador legado
│   └── improved_optimizer.py  # Otimizador melhorado
├── models/
│   ├── config.py          # Configurações do matching
│   └── person.py          # Classe Person
├── scripts/               # Scripts auxiliares
│   ├── split_by_gender.py # Separar dados por gênero
│   ├── run_split.py       # Script rápido de separação
│   └── analyze.py         # Análise dos dados
└── files/
    ├── data/              # Dados de entrada (Excel)
    └── results/           # Resultados do matching
```

## 🚀 Como Usar

### 1. Preparar os Dados

Primeiro, separe os dados por gênero:

```bash
python scripts/split_by_gender.py --input "./files/data/26.1/dados.xlsx" --women-file "./files/data/26.1/women_names.txt"
```

O arquivo `women_names.txt` deve conter um nome feminino por linha. Todos os outros serão considerados masculinos.

### 2. Executar o Matching

Para mulheres:
```bash
python match_improved.py --xlsx-path "./files/data/26.1/data_split.xlsx" --sheet "Women" --s m --to-excel --save-path "./files/results/26.1"
```

Para homens:
```bash
python match_improved.py --xlsx-path "./files/data/26.1/data_split.xlsx" --sheet "Men" --s h --to-excel --save-path "./files/results/26.1"
```

### 3. Argumentos Disponíveis

| Argumento | Descrição | Padrão |
|-----------|-----------|--------|
| `--xlsx-path` | Caminho do arquivo Excel | `data/dados.xlsx` |
| `--sheet` | Nome da aba no Excel | `Sheet1` |
| `--s` | Gênero (`m`=mulher, `h`=homem) | obrigatório |
| `--to-excel` | Salvar resultados em Excel | - |
| `--to-csv` | Salvar resultados em CSV | - |
| `--save-path` | Pasta para salvar resultados | `.` |
| `--comment-weight` | Peso para similaridade de comentários | `0.1` |
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
# 1. Criar arquivo com nomes das mulheres
echo "Maria Santos
Ana Paula
..." > women_names.txt

# 2. Separar dados por gênero
python scripts/run_split.py

# 3. Executar matching para mulheres
python match_improved.py --xlsx-path "./files/data/26.1/data_split.xlsx" --sheet "Women" --s m --to-excel --save-path "./files/results/26.1"

# 4. Executar matching para homens
python match_improved.py --xlsx-path "./files/data/26.1/data_split.xlsx" --sheet "Men" --s h --to-excel --save-path "./files/results/26.1"
```

## 📌 Notas Importantes

- A coluna de nomes deve ser `"Tell us what's you name: "` (ou será detectada automaticamente)
- A coluna de tipo deve conter `"Brazilian student (Buddy)"` ou `"International student (Incoming)"`
- Comentários triviais (`.`, `/`, emojis) são ignorados no cálculo de similaridade