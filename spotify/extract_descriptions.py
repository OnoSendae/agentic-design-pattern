#!/usr/bin/env python3
"""
Script para extrair e formatar descrições dos episódios do JSON
"""

import json
import re

def format_description(description):
    """Formata uma descrição convertendo \n para quebras de linha e organizando markdown"""
    
    # Remove \n e quebra em linhas
    lines = description.replace('\\n', '\n').split('\n')
    
    # Remove linhas vazias do início e fim
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append("")
            continue
            
        # Se é um título com emoji - mantém o emoji
        if line.startswith('🎯'):
            if formatted_lines and formatted_lines[-1]:
                formatted_lines.append("")
            # Remove ** e mantém o emoji
            clean_title = line.replace('**', '').strip()
            formatted_lines.append(f"# {clean_title}")
            formatted_lines.append("")
            
        elif line.startswith('🔍'):
            if formatted_lines and formatted_lines[-1]:
                formatted_lines.append("")
            clean_title = line.replace('**', '').strip()
            formatted_lines.append(f"## {clean_title}")
            formatted_lines.append("")
            
        elif line.startswith('💻') or line.startswith('📚') or line.startswith('⚖️') or line.startswith('📖') or line.startswith('👤'):
            if formatted_lines and formatted_lines[-1]:
                formatted_lines.append("")
            clean_title = line.replace('**', '').strip()
            formatted_lines.append(f"## {clean_title}")
            formatted_lines.append("")
            
        # Se é uma lista com bullet
        elif line.startswith('•'):
            # Adiciona quebra antes da primeira lista se necessário
            if formatted_lines and not formatted_lines[-1].startswith('-'):
                formatted_lines.append("")
            formatted_lines.append("- " + line[1:].strip())
            
        # Se é texto normal
        else:
            formatted_lines.append(line)
    
    # Processa links
    final_lines = []
    for line in formatted_lines:
        # Converte URLs em links markdown mais inteligentes
        if 'https://drive.google.com' in line and 'Google Drive' not in line:
            line = re.sub(r'https://drive\.google\.com/[^\s]+', '[Google Drive](https://drive.google.com/drive/u/0/folders/1Y3U3IrYCiJ3E45Z8okR5eCg7OPnWQtPV)', line)
        elif 'linkedin.com' in line and 'LinkedIn' not in line:
            line = re.sub(r'https://www\.linkedin\.com/[^\s]+', '[LinkedIn](https://www.linkedin.com/in/searchguy/)', line)
        elif 'linkedin.com' in line and 'LinkedIn' in line:
            line = re.sub(r'Antonio Gulli - LinkedIn: https://www\.linkedin\.com/[^\s]+', '**Antonio Gulli** - [LinkedIn](https://www.linkedin.com/in/searchguy/)', line)
        final_lines.append(line)
    
    return '\n'.join(final_lines)

def main():
    # Carrega o JSON
    with open('spotify-complete-metadata.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extrai descrições dos episódios principais
    main_episodes = []
    special_episodes = []
    
    for episode in data['episodes']:
        episode_info = {
            'number': episode['episode_number'],
            'title': episode['title'],
            'type': episode['episode_type'],
            'description': episode['description']
        }
        
        if episode['episode_type'] == 'main':
            main_episodes.append(episode_info)
        else:
            special_episodes.append(episode_info)
    
    # Ordena episódios principais numericamente
    main_episodes.sort(key=lambda x: int(x['number']))
    
    # Gera markdown para episódios principais
    main_content = ["# Descrições dos Episódios - Padrões de Design Agênticos\n"]
    
    for episode in main_episodes:
        # Limpa o título removendo o prefixo [Episódio XX]
        clean_title = episode['title'].replace(f'[Episódio {episode["number"]}] ', '')
        main_content.append(f"## Episódio {episode['number']} - {clean_title}")
        main_content.append("")
        main_content.append(format_description(episode['description']))
        main_content.append("")
        main_content.append("---")
        main_content.append("")
    
    # Gera markdown para episódios especiais
    special_content = ["# Apêndices - Padrões de Design Agênticos\n"]
    
    for episode in special_episodes:
        # Limpa o título removendo o prefixo [Episódio X]
        clean_title = episode['title'].replace(f'[Episódio {episode["number"]}] ', '')
        special_content.append(f"## Apêndice {episode['number']} - {clean_title}")
        special_content.append("")
        special_content.append(format_description(episode['description']))
        special_content.append("")
        special_content.append("---")
        special_content.append("")
    
    # Salva os arquivos
    with open('episodes-descriptions-formatted.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(main_content))
    
    with open('episodes-special-descriptions.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(special_content))
    
    print(f"✅ Extraídas {len(main_episodes)} descrições de episódios principais")
    print(f"✅ Extraídas {len(special_episodes)} descrições de episódios especiais")
    print("📁 Arquivos criados:")
    print("   - episodes-descriptions-formatted.md")
    print("   - episodes-special-descriptions.md")

if __name__ == "__main__":
    main()
