#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { google } = require('googleapis');
const readline = require('readline');

// Configuration
const YOUTUBE_VIDEOS_DIR = './youtube_videos';
const METADATA_FILE = './youtube_videos/youtube_metadata.json';
const SCOPES = ['https://www.googleapis.com/auth/youtube.upload'];
const TOKEN_PATH = './youtube_token.json';
const CREDENTIALS_PATH = './youtube_credentials.json';

// YouTube API configuration
const YOUTUBE_API_VERSION = 'v3';

class YouTubeUploader {
    constructor() {
        this.youtube = null;
        this.auth = null;
    }

    // Load credentials from file
    async loadCredentials() {
        try {
            const credentials = JSON.parse(fs.readFileSync(CREDENTIALS_PATH, 'utf8'));
            return credentials;
        } catch (error) {
            console.error('❌ Erro ao carregar credenciais:', error.message);
            console.log('\n📝 Certifique-se de que o arquivo youtube_credentials.json existe');
            console.log('   e contém as credenciais da API do YouTube.');
            process.exit(1);
        }
    }

    // Create OAuth2 client
    async createOAuth2Client(credentials) {
        const { client_secret, client_id, redirect_uris } = credentials.installed || credentials.web;
        const oAuth2Client = new google.auth.OAuth2(client_id, client_secret, redirect_uris[0]);

        // Check if we have previously stored a token
        try {
            const token = JSON.parse(fs.readFileSync(TOKEN_PATH, 'utf8'));
            oAuth2Client.setCredentials(token);
            return oAuth2Client;
        } catch (error) {
            return await this.getNewToken(oAuth2Client);
        }
    }

    // Get new token after prompting for user authorization
    async getNewToken(oAuth2Client) {
        const authUrl = oAuth2Client.generateAuthUrl({
            access_type: 'offline',
            scope: SCOPES,
        });

        console.log('\n🔐 Autorização necessária!');
        console.log('📋 Siga estes passos:');
        console.log('1. Abra este link no seu navegador:');
        console.log(`   ${authUrl}`);
        console.log('2. Faça login na sua conta do YouTube');
        console.log('3. Autorize o aplicativo');
        console.log('4. Copie o código de autorização');
        console.log('');

        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
        });

        return new Promise((resolve, reject) => {
            rl.question('🔑 Cole o código de autorização aqui: ', async (code) => {
                rl.close();
                try {
                    const { tokens } = await oAuth2Client.getToken(code);
                    oAuth2Client.setCredentials(tokens);

                    // Store the token for future use
                    fs.writeFileSync(TOKEN_PATH, JSON.stringify(tokens));
                    console.log('✅ Token salvo em:', TOKEN_PATH);

                    resolve(oAuth2Client);
                } catch (error) {
                    console.error('❌ Erro ao obter token:', error.message);
                    reject(error);
                }
            });
        });
    }

    // Initialize YouTube API
    async initialize() {
        console.log('🚀 Inicializando YouTube API...');

        const credentials = await this.loadCredentials();
        this.auth = await this.createOAuth2Client(credentials);
        this.youtube = google.youtube({ version: YOUTUBE_API_VERSION, auth: this.auth });

        console.log('✅ YouTube API inicializada com sucesso!');
    }

    // Load metadata from file
    loadMetadata() {
        try {
            const metadata = JSON.parse(fs.readFileSync(METADATA_FILE, 'utf8'));
            return metadata;
        } catch (error) {
            console.error('❌ Erro ao carregar metadados:', error.message);
            console.log('📝 Execute primeiro o script generate_youtube_videos.js');
            process.exit(1);
        }
    }

    // Upload single video
    async uploadVideo(videoPath, metadata) {
        const videoTitle = metadata.youtube_title;
        const videoDescription = metadata.youtube_description;
        const tags = metadata.tags;

        console.log(`\n📤 Enviando: ${videoTitle}`);
        console.log(`📁 Arquivo: ${path.basename(videoPath)}`);

        try {
            const response = await this.youtube.videos.insert({
                part: 'snippet,status',
                requestBody: {
                    snippet: {
                        title: videoTitle,
                        description: videoDescription,
                        tags: tags,
                        categoryId: '28', // Science & Technology
                        defaultLanguage: 'pt',
                        defaultAudioLanguage: 'pt'
                    },
                    status: {
                        privacyStatus: 'private', // Start as private, you can change later
                        selfDeclaredMadeForKids: false
                    }
                },
                media: {
                    body: fs.createReadStream(videoPath)
                }
            });

            const videoId = response.data.id;
            const videoUrl = `https://www.youtube.com/watch?v=${videoId}`;

            console.log(`✅ Upload concluído!`);
            console.log(`🔗 URL: ${videoUrl}`);
            console.log(`🆔 Video ID: ${videoId}`);

            return {
                success: true,
                videoId,
                videoUrl,
                title: videoTitle
            };

        } catch (error) {
            console.error(`❌ Erro no upload:`, error.message);
            return {
                success: false,
                error: error.message,
                title: videoTitle
            };
        }
    }

    // Upload all videos
    async uploadAllVideos() {
        console.log('📊 Carregando metadados...');
        const metadata = this.loadMetadata();

        if (!metadata.episodes || metadata.episodes.length === 0) {
            console.error('❌ Nenhum episódio encontrado nos metadados!');
            return;
        }

        console.log(`📈 Encontrados ${metadata.episodes.length} episódios para upload`);

        const results = [];
        let successCount = 0;
        let errorCount = 0;

        for (let i = 0; i < metadata.episodes.length; i++) {
            const episode = metadata.episodes[i];
            const videoPath = path.join(YOUTUBE_VIDEOS_DIR, episode.video_file);

            console.log(`\n📈 Progresso: ${i + 1}/${metadata.episodes.length}`);

            // Check if video file exists
            if (!fs.existsSync(videoPath)) {
                console.error(`❌ Arquivo não encontrado: ${episode.video_file}`);
                results.push({
                    success: false,
                    error: 'Arquivo não encontrado',
                    title: episode.title
                });
                errorCount++;
                continue;
            }

            const result = await this.uploadVideo(videoPath, episode);
            results.push(result);

            if (result.success) {
                successCount++;
            } else {
                errorCount++;
            }

            // Add delay between uploads to avoid rate limiting
            if (i < metadata.episodes.length - 1) {
                console.log('⏳ Aguardando 5 segundos antes do próximo upload...');
                await new Promise(resolve => setTimeout(resolve, 5000));
            }
        }

        return { results, successCount, errorCount };
    }

    // Generate upload report
    generateReport(uploadResults) {
        const { results, successCount, errorCount } = uploadResults;

        const report = {
            upload_date: new Date().toISOString(),
            total_episodes: results.length,
            successful_uploads: successCount,
            failed_uploads: errorCount,
            videos: results.map(result => ({
                title: result.title,
                success: result.success,
                video_id: result.videoId || null,
                video_url: result.videoUrl || null,
                error: result.error || null
            }))
        };

        const reportPath = path.join(YOUTUBE_VIDEOS_DIR, 'upload_report.json');
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

        console.log('\n' + '='.repeat(60));
        console.log('📊 RELATÓRIO DE UPLOAD');
        console.log('='.repeat(60));
        console.log(`✅ Uploads bem-sucedidos: ${successCount}`);
        console.log(`❌ Uploads com erro: ${errorCount}`);
        console.log(`📋 Relatório salvo em: ${reportPath}`);

        if (errorCount > 0) {
            console.log('\n❌ Vídeos com erro:');
            results.filter(r => !r.success).forEach(r => {
                console.log(`   - ${r.title}: ${r.error}`);
            });
        }

        if (successCount > 0) {
            console.log('\n✅ Vídeos enviados com sucesso:');
            results.filter(r => r.success).forEach(r => {
                console.log(`   - ${r.title}`);
                console.log(`     🔗 ${r.videoUrl}`);
            });
        }
    }
}

// Main execution function
async function main() {
    console.log('🎬 YouTube Uploader - Iniciando...\n');

    // Check if required files exist
    if (!fs.existsSync(METADATA_FILE)) {
        console.error('❌ Arquivo de metadados não encontrado!');
        console.log('📝 Execute primeiro: node generate_youtube_videos.js');
        process.exit(1);
    }

    if (!fs.existsSync(CREDENTIALS_PATH)) {
        console.error('❌ Arquivo de credenciais não encontrado!');
        console.log('📝 Crie o arquivo youtube_credentials.json com suas credenciais da API');
        process.exit(1);
    }

    try {
        const uploader = new YouTubeUploader();
        await uploader.initialize();

        console.log('\n🚀 Iniciando upload dos vídeos...');
        const uploadResults = await uploader.uploadAllVideos();

        uploader.generateReport(uploadResults);

        console.log('\n🎉 Processo de upload concluído!');
        console.log('\n📝 Próximos passos:');
        console.log('1. Verifique os vídeos no YouTube Studio');
        console.log('2. Ajuste configurações de privacidade se necessário');
        console.log('3. Adicione thumbnails personalizados');
        console.log('4. Configure agendamento de publicação');

    } catch (error) {
        console.error('❌ Erro fatal:', error.message);
        process.exit(1);
    }
}

// Run the script
if (require.main === module) {
    main();
}

module.exports = YouTubeUploader;
