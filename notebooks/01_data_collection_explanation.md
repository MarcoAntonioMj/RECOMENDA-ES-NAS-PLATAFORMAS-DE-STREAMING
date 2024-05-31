# Coleta de Dados

### Fonte dos Dados

Os dados utilizados neste projeto são provenientes da API pública fictícia disponível em [https://guilhermeonrails.github.io/api-csharp-songs/songs.json](https://guilhermeonrails.github.io/api-csharp-songs/songs.json). Estes dados são gerados artificialmente para fins educacionais e não contêm informações pessoais ou sensíveis.

### Estrutura dos Dados

Cada registro na API contém informações sobre uma música, incluindo:

- `id`: Identificador único da música.
- `title`: Título da música.
- `artist`: Artista que performa a música.
- `genre`: Gênero musical.
- `year`: Ano de lançamento.

### Coleta de Dados

Nesta etapa, realizamos uma requisição à API, visualizamos uma amostra dos dados e salvamos os dados brutos em um arquivo JSON no diretório `data/raw/`.
