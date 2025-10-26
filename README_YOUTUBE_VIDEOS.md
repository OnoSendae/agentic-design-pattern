# 🎬 Gerador e Uploader de Vídeos para YouTube

Este projeto converte todos os episódios de podcast em vídeos para YouTube com uma imagem fixa de fundo e faz upload automático via API.

## 📋 Pré-requisitos

### 1. Node.js
- **Versão**: Node.js 14+ 
- **Instalação**: [Download do site oficial](https://nodejs.org/)

### 2. FFmpeg
- **Status**: ✅ Já instalado (conforme informado)
- **Verificação**: Execute `ffmpeg -version` no terminal

### 3. Dependências do Sistema
- **macOS**: Já incluído no sistema
- **Linux**: `sudo apt-get install ffmpeg` (Ubuntu/Debian)
- **Windows**: Download do [FFmpeg](https://ffmpeg.org/download.html)

### 4. YouTube API (para upload automático)
- **Conta do Google/YouTube**: Necessária
- **Google Cloud Console**: Para configurar API
- **Credenciais**: Arquivo `youtube_credentials.json`

## 🚀 Como Usar

### 1. Instalar Dependências
```bash
# Navegue até o diretório do projeto
cd /Users/cleberdasilvahensel/Desktop/source/ono-sendae/ia-book

# Instalar dependências
npm install
```

### 2. Gerar Vídeos
```bash
# Gerar vídeos com imagem fixa
npm run generate
# ou
node generate_youtube_videos.js
```

### 3. Configurar YouTube API (primeira vez)
```bash
# Siga o guia completo
open YOUTUBE_API_SETUP.md
```

### 4. Fazer Upload Automático
```bash
# Upload via API do YouTube
npm run upload
# ou
node youtube_uploader.js
```

### 5. Processo Completo
```bash
# Gerar vídeos + Upload automático
npm run full-process
```

## 📁 Arquivos do Projeto

### Scripts Principais
- **`generate_youtube_videos.js`** - Gera vídeos com imagem fixa
- **`youtube_uploader.js`** - Upload automático via API
- **`package.json`** - Dependências e scripts

### Documentação
- **`README_YOUTUBE_VIDEOS.md`** - Este arquivo
- **`YOUTUBE_API_SETUP.md`** - Guia de configuração da API

### O que os Scripts Fazem

#### Gerador de Vídeos (`generate_youtube_videos.js`)
- ✅ Lê todos os arquivos `.mp3` da pasta `podcasts/`
- ✅ Usa a imagem `logo-agentic.png` como fundo fixo
- ✅ Gera vídeos em resolução 1920x1080 (Full HD)
- ✅ Cria pasta `youtube_videos/` com todos os vídeos
- ✅ Gera arquivo `youtube_metadata.json` com metadados para upload

#### Uploader (`youtube_uploader.js`)
- ✅ Autentica com YouTube API
- ✅ Faz upload automático de todos os vídeos
- ✅ Configura metadados (título, descrição, tags)
- ✅ Gera relatório de upload
- ✅ Trata erros e rate limiting

### 3. Estrutura de Saída
```
youtube_videos/
├── episodio_01_prompt_chaining_como_o_encadeamento_de_prompts_e_a_engenharia.mp4
├── episodio_02_roteamento_em_ia_o_segredo_para_agentes_inteligentes_que_decid.mp4
├── episodio_03_paralelizacao_em_agentes_de_ia_o_segredo_para_sistemas_mais_ra.mp4
├── ...
└── youtube_metadata.json
```

## 📊 Configurações do Vídeo

- **Resolução**: 1920x1080 (Full HD)
- **Frame Rate**: 30 FPS
- **Codec de Vídeo**: H.264 (libx264)
- **Codec de Áudio**: AAC
- **Bitrate de Áudio**: 192k
- **Formato de Pixel**: yuv420p (compatível com YouTube)

## 📝 Metadados Gerados

O arquivo `youtube_metadata.json` contém:
- Títulos otimizados para YouTube
- Descrições com hashtags
- Tags relevantes
- Números dos episódios
- Nomes dos arquivos de vídeo

## ⚙️ Personalização

### Alterar Resolução
Edite a linha no script:
```javascript
const VIDEO_RESOLUTION = '1280x720'; // HD
// ou
const VIDEO_RESOLUTION = '3840x2160'; // 4K
```

### Alterar Frame Rate
```javascript
const VIDEO_FPS = 24; // 24 FPS
// ou
const VIDEO_FPS = 60; // 60 FPS
```

### Alterar Imagem de Fundo
```javascript
const BACKGROUND_IMAGE = './caminho/para/sua/imagem.png';
```

## 🔧 Solução de Problemas

### Erro: "ffmpeg not found"
```bash
# Verificar se FFmpeg está instalado
ffmpeg -version

# Se não estiver, instalar:
# macOS (com Homebrew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### Erro: "Permission denied"
```bash
# Dar permissão de execução ao script
chmod +x generate_youtube_videos.js
```

### Erro: "No such file or directory"
- Verifique se está no diretório correto
- Confirme que a pasta `podcasts/` existe
- Confirme que `logo-agentic.png` existe

## 📈 Monitoramento do Progresso

O script mostra:
- ✅ Progresso em tempo real
- 📊 Contador de sucessos/erros
- 📝 Títulos dos episódios processados
- ⚙️ Comandos FFmpeg executados

## 🎯 Próximos Passos

1. **Verificar Vídeos**: Abra alguns vídeos gerados para verificar qualidade
2. **Upload no YouTube**: Use o arquivo de metadados para facilitar o upload
3. **Personalizar**: Ajuste títulos e descrições conforme necessário
4. **Thumbnails**: Considere criar thumbnails personalizados

## 📞 Suporte

Se encontrar problemas:
1. Verifique se todos os pré-requisitos estão instalados
2. Confirme que os arquivos de podcast estão na pasta correta
3. Execute o script com `node generate_youtube_videos.js` e veja as mensagens de erro

---

**Tempo estimado**: ~2-5 minutos por episódio (dependendo do tamanho do arquivo de áudio)
