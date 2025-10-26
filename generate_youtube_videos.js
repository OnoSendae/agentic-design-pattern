#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Configuration
const PODCASTS_DIR = './podcasts';
const OUTPUT_DIR = './youtube_videos';
const BACKGROUND_IMAGE = './podcasts/logo-agentic.png';
const VIDEO_RESOLUTION = '1920x1080'; // Full HD
const VIDEO_FPS = 30;

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Get all MP3 files from podcasts directory
function getPodcastFiles() {
    const files = fs.readdirSync(PODCASTS_DIR);
    return files.filter(file => file.endsWith('.mp3') && file !== 'logo-agentic.png');
}

// Extract episode number and title from filename
function parseFilename(filename) {
    const nameWithoutExt = path.basename(filename, '.mp3');

    // Handle numbered episodes (01-, 02-, etc.)
    const numberedMatch = nameWithoutExt.match(/^(\d+)-(.+)$/);
    if (numberedMatch) {
        return {
            episodeNumber: parseInt(numberedMatch[1]),
            title: numberedMatch[2].replace(/_/g, ' ').replace(/__/g, ': ')
        };
    }

    // Handle lettered episodes (a-, b-, etc.)
    const letteredMatch = nameWithoutExt.match(/^([a-z])-(.+)$/);
    if (letteredMatch) {
        return {
            episodeNumber: letteredMatch[1].charCodeAt(0) - 96, // a=1, b=2, etc.
            title: letteredMatch[2].replace(/_/g, ' ').replace(/__/g, ': ')
        };
    }

    // Fallback
    return {
        episodeNumber: 0,
        title: nameWithoutExt.replace(/_/g, ' ').replace(/__/g, ': ')
    };
}

// Generate video for a single podcast
function generateVideo(podcastFile) {
    const inputPath = path.join(PODCASTS_DIR, podcastFile);
    const { episodeNumber, title } = parseFilename(podcastFile);
    const outputFilename = `episodio_${episodeNumber.toString().padStart(2, '0')}_${title.replace(/[^a-zA-Z0-9\s]/g, '').replace(/\s+/g, '_').toLowerCase()}.mp4`;
    const outputPath = path.join(OUTPUT_DIR, outputFilename);

    console.log(`\n🎬 Processando: ${podcastFile}`);
    console.log(`📝 Título: ${title}`);
    console.log(`📁 Saída: ${outputFilename}`);

    try {
        // FFmpeg command to create video with static image and audio
        const ffmpegCommand = [
            'ffmpeg',
            '-y', // Overwrite output file
            '-loop 1', // Loop the input image
            `-i "${BACKGROUND_IMAGE}"`, // Background image
            `-i "${inputPath}"`, // Audio file
            '-c:v libx264', // Video codec
            '-tune stillimage', // Optimize for still images
            '-c:a aac', // Audio codec
            '-b:a 192k', // Audio bitrate
            '-pix_fmt yuv420p', // Pixel format for compatibility
            '-shortest', // End when shortest input ends
            `-s ${VIDEO_RESOLUTION}`, // Video resolution
            `-r ${VIDEO_FPS}`, // Frame rate
            `"${outputPath}"`
        ].join(' ');

        console.log(`⚙️  Executando FFmpeg...`);
        execSync(ffmpegCommand, { stdio: 'inherit' });

        console.log(`✅ Vídeo gerado com sucesso: ${outputFilename}`);
        return { success: true, outputPath, title, episodeNumber };

    } catch (error) {
        console.error(`❌ Erro ao processar ${podcastFile}:`, error.message);
        return { success: false, error: error.message, podcastFile };
    }
}

// Generate metadata file for YouTube upload
function generateMetadataFile(results) {
    const successfulResults = results.filter(r => r.success);

    const metadata = {
        generated_at: new Date().toISOString(),
        total_episodes: successfulResults.length,
        episodes: successfulResults.map(result => ({
            episode_number: result.episodeNumber,
            title: result.title,
            video_file: path.basename(result.outputPath),
            youtube_title: `Episódio ${result.episodeNumber}: ${result.title}`,
            youtube_description: `🎧 Episódio ${result.episodeNumber} do podcast sobre Agentes de IA\n\n${result.title}\n\n#IA #AgentesDeIA #InteligenciaArtificial #Podcast #Tecnologia`,
            tags: ['IA', 'Agentes de IA', 'Inteligência Artificial', 'Podcast', 'Tecnologia', 'Machine Learning']
        }))
    };

    const metadataPath = path.join(OUTPUT_DIR, 'youtube_metadata.json');
    fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));

    console.log(`\n📋 Metadados gerados: ${metadataPath}`);
    return metadataPath;
}

// Main execution
function main() {
    console.log('🚀 Iniciando geração de vídeos para YouTube...\n');

    // Check if background image exists
    if (!fs.existsSync(BACKGROUND_IMAGE)) {
        console.error(`❌ Imagem de fundo não encontrada: ${BACKGROUND_IMAGE}`);
        process.exit(1);
    }

    // Get all podcast files
    const podcastFiles = getPodcastFiles();

    if (podcastFiles.length === 0) {
        console.error('❌ Nenhum arquivo de podcast encontrado!');
        process.exit(1);
    }

    console.log(`📊 Encontrados ${podcastFiles.length} episódios de podcast`);
    console.log(`🖼️  Imagem de fundo: ${BACKGROUND_IMAGE}`);
    console.log(`📁 Diretório de saída: ${OUTPUT_DIR}\n`);

    // Process each podcast
    const results = [];
    let successCount = 0;
    let errorCount = 0;

    for (let i = 0; i < podcastFiles.length; i++) {
        const podcastFile = podcastFiles[i];
        console.log(`\n📈 Progresso: ${i + 1}/${podcastFiles.length}`);

        const result = generateVideo(podcastFile);
        results.push(result);

        if (result.success) {
            successCount++;
        } else {
            errorCount++;
        }
    }

    // Generate metadata file
    const metadataPath = generateMetadataFile(results);

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('📊 RESUMO DA GERAÇÃO');
    console.log('='.repeat(60));
    console.log(`✅ Sucessos: ${successCount}`);
    console.log(`❌ Erros: ${errorCount}`);
    console.log(`📁 Vídeos salvos em: ${OUTPUT_DIR}`);
    console.log(`📋 Metadados: ${metadataPath}`);

    if (errorCount > 0) {
        console.log('\n❌ Episódios com erro:');
        results.filter(r => !r.success).forEach(r => {
            console.log(`   - ${r.podcastFile}`);
        });
    }

    console.log('\n🎉 Processo concluído!');
    console.log('\n📝 Próximos passos:');
    console.log('1. Verifique os vídeos gerados na pasta youtube_videos/');
    console.log('2. Use o arquivo youtube_metadata.json para upload no YouTube');
    console.log('3. Ajuste títulos e descrições conforme necessário');
}

// Run the script
if (require.main === module) {
    main();
}

module.exports = { generateVideo, parseFilename, getPodcastFiles };
