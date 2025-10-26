# 🔐 Configuração da API do YouTube

Este guia te ajudará a configurar a API do YouTube para upload automático dos vídeos.

## 📋 Pré-requisitos

### 1. Conta do Google/YouTube
- ✅ Conta do Google ativa
- ✅ Canal do YouTube criado
- ✅ Verificação de telefone (se necessário)

### 2. Google Cloud Console
- Acesso ao [Google Cloud Console](https://console.cloud.google.com/)

## 🚀 Passo a Passo

### Passo 1: Criar Projeto no Google Cloud Console

1. **Acesse**: [Google Cloud Console](https://console.cloud.google.com/)
2. **Crie um novo projeto**:
   - Clique em "Selecionar projeto" no topo
   - Clique em "NOVO PROJETO"
   - Nome: `YouTube Uploader` (ou qualquer nome)
   - Clique em "CRIAR"

### Passo 2: Habilitar YouTube Data API v3

1. **No menu lateral**, vá em "APIs e serviços" > "Biblioteca"
2. **Pesquise por**: "YouTube Data API v3"
3. **Clique em**: "YouTube Data API v3"
4. **Clique em**: "HABILITAR"

### Passo 3: Criar Credenciais

1. **Vá em**: "APIs e serviços" > "Credenciais"
2. **Clique em**: "+ CRIAR CREDENCIAIS"
3. **Selecione**: "ID do cliente OAuth 2.0"
4. **Tipo de aplicativo**: "Aplicativo de desktop"
5. **Nome**: `YouTube Uploader`
6. **Clique em**: "CRIAR"

### Passo 4: Baixar Credenciais

1. **Após criar**, clique no ícone de download (⬇️)
2. **Salve o arquivo** como `youtube_credentials.json`
3. **Mova o arquivo** para a pasta do projeto:
   ```bash
   mv ~/Downloads/youtube_credentials.json /Users/cleberdasilvahensel/Desktop/source/ono-sendae/ia-book/
   ```

### Passo 5: Configurar Tela de Consentimento

1. **Vá em**: "APIs e serviços" > "Tela de consentimento OAuth"
2. **Tipo de usuário**: "Externo" (se não for organização)
3. **Preencha os campos obrigatórios**:
   - Nome do aplicativo: `YouTube Uploader`
   - Email de suporte: seu email
   - Email de contato do desenvolvedor: seu email
4. **Clique em**: "SALVAR E CONTINUAR"
5. **Escopo**: Clique em "ADICIONAR OU REMOVER ESCOPOS"
   - Procure por: "YouTube Data API v3"
   - Selecione: `https://www.googleapis.com/auth/youtube.upload`
6. **Usuários de teste**: Adicione seu email
7. **Clique em**: "SALVAR E CONTINUAR"

## 📦 Instalação das Dependências

### 1. Instalar Google APIs Client
```bash
npm install googleapis
```

### 2. Verificar Estrutura de Arquivos
```
/Users/cleberdasilvahensel/Desktop/source/ono-sendae/ia-book/
├── youtube_credentials.json  ← Suas credenciais
├── youtube_uploader.js       ← Script de upload
├── generate_youtube_videos.js ← Script de geração
└── youtube_videos/           ← Pasta com vídeos
```

## 🎬 Como Usar

### 1. Gerar Vídeos (se ainda não fez)
```bash
node generate_youtube_videos.js
```

### 2. Fazer Upload
```bash
node youtube_uploader.js
```

### 3. Primeira Execução
Na primeira vez, o script vai:
1. Abrir uma URL no navegador
2. Pedir autorização
3. Gerar um código
4. Salvar o token para uso futuro

## ⚙️ Configurações do Upload

### Configurações Padrão
- **Privacidade**: Privado (você pode mudar depois)
- **Categoria**: Ciência e Tecnologia
- **Idioma**: Português
- **Crianças**: Não

### Personalizar Configurações
Edite o arquivo `youtube_uploader.js`:

```javascript
// Linha ~100 - Configurações do vídeo
status: {
    privacyStatus: 'unlisted', // ou 'public', 'private'
    selfDeclaredMadeForKids: false
},
snippet: {
    categoryId: '28', // 28 = Science & Technology
    defaultLanguage: 'pt',
    defaultAudioLanguage: 'pt'
}
```

## 🔧 Solução de Problemas

### Erro: "Quota exceeded"
- **Causa**: Limite diário da API atingido
- **Solução**: Aguarde 24h ou solicite aumento de quota

### Erro: "Invalid credentials"
- **Causa**: Arquivo de credenciais incorreto
- **Solução**: Verifique se `youtube_credentials.json` está correto

### Erro: "Access blocked"
- **Causa**: Aplicativo não verificado
- **Solução**: Use apenas com sua conta por enquanto

### Erro: "File not found"
- **Causa**: Vídeos não foram gerados
- **Solução**: Execute primeiro `node generate_youtube_videos.js`

## 📊 Limites da API

### Quotas Diárias
- **Uploads**: 6 vídeos por dia (padrão)
- **Requests**: 10.000 por dia
- **Aumento**: Solicite no Google Cloud Console

### Tamanho de Arquivo
- **Máximo**: 256 GB
- **Recomendado**: < 2 GB para uploads mais rápidos

## 🎯 Próximos Passos

1. **Teste com 1 vídeo** primeiro
2. **Verifique no YouTube Studio**
3. **Configure agendamento** se desejar
4. **Adicione thumbnails** personalizados
5. **Monitore o relatório** de upload

## 📞 Suporte

### Links Úteis
- [YouTube Data API Documentation](https://developers.google.com/youtube/v3)
- [Google Cloud Console](https://console.cloud.google.com/)
- [YouTube Studio](https://studio.youtube.com/)

### Comandos de Verificação
```bash
# Verificar se as dependências estão instaladas
npm list googleapis

# Verificar se os arquivos existem
ls -la youtube_credentials.json
ls -la youtube_videos/
```

---

**⚠️ Importante**: Mantenha suas credenciais seguras e nunca as compartilhe publicamente!
