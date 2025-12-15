# Buddies x International Students Matching

## Usage:


```
python main.py -h
```

Assim você terá todos os argumentos permitidos.

### Example


- `--xlsx-path` : Caminho para o arquivo Excel com os dados
- `--sheet` : Nome da aba no arquivo Excel
- `--to-csv` : Salvar resultados em CSV
- `--to-excel` : Salvar resultados em Excel
- `--save-path` : Diretório para salvar os resultados
- `--s` : Especificar o gênero (m ou h)
- `--classes` : Especificar classes | default: ['buddy', 'int']

```
python match.py --xlsx-path .\files\data\BuddyProgramMatching.xlsx --sheet "Buddy Girls" --to-excel --save-path .\files\results --s "m"
```

## !!! AVISO !!!

Por enquanto funciona apenas para colunas com 2 classes diferentes, como "Buddies" e "Internacional Students".