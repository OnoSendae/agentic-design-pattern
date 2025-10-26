#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Configuration
const PODCASTS_DIR = './podcasts';
const OUTPUT_DIR = './test_video';
const BACKGROUND_IMAGE = './podcasts/logo-agentic.png';
const VIDEO_RESOLUTION = '1920x1080'; // Full HD
const VIDEO_FPS = 30;

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Get the first MP3 file for testing
function getTestPodcastFile() {
    const files = fs.readdirSync(PODCASTS_DIR);
    const mp3Files = files.filter(file => file.endsWith('.mp3') && file !== 'logo-agentic.png');

    if (mp3Files.length === 0) {
        throw new Error('Nenhum arquivo de podcast encontrado!');
    }

    // Return the first file (usually the shortest for testing)
    return mp3Files[0];
}

// Extract episode info from filename
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

// Generate test video
function generateTestVideo(podcastFile) {
    const inputPath = path.join(PODCASTS_DIR, podcastFile);
    const { episodeNumber, title } = parseFilename(podcastFile);
    const outputFilename = `test_episodio_${episodeNumber.toString().padStart(2, '0')}.mp4`;
    const outputPath = path.join(OUTPUT_DIR, outputFilename);

    console.log(`\n🎬 Gerando vídeo de teste...`);
    console.log(`📁 Arquivo: ${podcastFile}`);
    console.log(`📝 Título: ${title}`);
    console.log(`📁 Saída: ${outputFilename}`);
    console.log(`🖼️  Imagem: ${BACKGROUND_IMAGE}`);
    console.log(`📐 Resolução: ${VIDEO_RESOLUTION}`);
    console.log(`🎞️  FPS: ${VIDEO_FPS}`);

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

        console.log(`\n⚙️  Executando FFmpeg...`);
        console.log(`🔧 Comando: ${ffmpegCommand.replace(/\s+/g, ' ')}`);

        const startTime = Date.now();
        execSync(ffmpegCommand, { stdio: 'inherit' });
        const endTime = Date.now();
        const duration = Math.round((endTime - startTime) / 1000);

        // Get file size
        const stats = fs.statSync(outputPath);
        const fileSizeMB = (stats.size / (1024 * 1024)).toFixed(2);

        console.log(`\n✅ Vídeo de teste gerado com sucesso!`);
        console.log(`📁 Arquivo: ${outputPath}`);
        console.log(`📊 Tamanho: ${fileSizeMB} MB`);
        console.log(`⏱️  Tempo de processamento: ${duration} segundos`);

        return {
            success: true,
            outputPath,
            title,
            episodeNumber,
            fileSizeMB,
            duration
        };

    } catch (error) {
        console.error(`❌ Erro ao processar ${podcastFile}:`, error.message);
        return { success: false, error: error.message, podcastFile };
    }
}

// Main execution
function main() {
    console.log('🧪 Gerador de Vídeo de Teste');
    console.log('='.repeat(50));

    // Check if background image exists
    if (!fs.existsSync(BACKGROUND_IMAGE)) {
        console.error(`❌ Imagem de fundo não encontrada: ${BACKGROUND_IMAGE}`);
        process.exit(1);
    }

    // Check if podcasts directory exists
    if (!fs.existsSync(PODCASTS_DIR)) {
        console.error(`❌ Diretório de podcasts não encontrado: ${PODCASTS_DIR}`);
        process.exit(1);
    }

    try {
        // Get test file
        const testFile = getTestPodcastFile();
        console.log(`📊 Arquivo selecionado para teste: ${testFile}`);

        // Generate test video
        const result = generateTestVideo(testFile);

        if (result.success) {
            console.log('\n' + '='.repeat(50));
            console.log('🎉 TESTE CONCLUÍDO COM SUCESSO!');
            console.log('='.repeat(50));
            console.log(`✅ Vídeo gerado: ${result.outputPath}`);
            console.log(`📊 Tamanho: ${result.fileSizeMB} MB`);
            console.log(`⏱️  Tempo: ${result.duration} segundos`);
            console.log(`📝 Título: ${result.title}`);

            console.log('\n📋 Próximos passos:');
            console.log('1. Abra o vídeo gerado para verificar a qualidade');
            console.log('2. Verifique se a imagem de fundo está correta');
            console.log('3. Teste o áudio e sincronização');
            console.log('4. Se estiver satisfeito, execute: node generate_youtube_videos.js');
            console.log('5. Para processar todos os episódios');

            console.log('\n🔧 Para ajustar configurações:');
            console.log('- Edite as constantes no início do arquivo');
            console.log('- Resolução: VIDEO_RESOLUTION');
            console.log('- FPS: VIDEO_FPS');
            console.log('- Imagem: BACKGROUND_IMAGE');

        } else {
            console.log('\n❌ TESTE FALHOU');
            console.log(`Erro: ${result.error}`);
            console.log('\n🔧 Verifique:');
            console.log('1. Se o FFmpeg está instalado: ffmpeg -version');
            console.log('2. Se os arquivos de podcast existem');
            console.log('3. Se a imagem de fundo existe');
        }

    } catch (error) {
        console.error('❌ Erro fatal:', error.message);
        process.exit(1);
    }
}

// Run the script
if (require.main === module) {
    main();
}

module.exports = { generateTestVideo, parseFilename, getTestPodcastFile };

