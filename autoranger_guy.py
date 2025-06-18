import os
import shutil
import re
import zipfile
import streamlit as st
import pandas as pd
from pathlib import Path
import platform
from datetime import datetime
from collections import Counter, defaultdict

# Configuration pour différents systèmes
def get_system_paths():
    """Retourne les chemins système selon l'OS"""
    system = platform.system()
    if system == "Windows":
        return {
            "desktop": os.path.join(os.path.expanduser("~"), "Desktop"),
            "documents": os.path.join(os.path.expanduser("~"), "Documents"),
            "downloads": os.path.join(os.path.expanduser("~"), "Downloads"),
            "separator": "\\"
        }
    else:  # macOS et Linux
        return {
            "desktop": os.path.join(os.path.expanduser("~"), "Desktop"),
            "documents": os.path.join(os.path.expanduser("~"), "Documents"),
            "downloads": os.path.join(os.path.expanduser("~"), "Downloads"),
            "separator": "/"
        }

def scan_directory_universal(directory, max_depth=3):
    """Scan universel avec gestion des permissions et profondeur limitée"""
    exclude_dirs = {
        '__pycache__', '.git', 'node_modules', '.vscode', '.idea',
        'System Volume Information', '$Recycle.Bin', '.Trash',
        '.DS_Store', 'Thumbs.db'
    }
    
    file_list = []
    error_list = []
    
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            return [], [f"Le répertoire {directory} n'existe pas"]
        
        def scan_recursive(path, current_depth=0):
            if current_depth > max_depth:
                return
            
            try:
                for item in path.iterdir():
                    if item.name.startswith('.') and item.name in exclude_dirs:
                        continue
                    
                    if item.is_file():
                        file_list.append(str(item))
                    elif item.is_dir() and item.name not in exclude_dirs:
                        scan_recursive(item, current_depth + 1)
            except PermissionError:
                error_list.append(f"Permission refusée pour: {path}")
            except Exception as e:
                error_list.append(f"Erreur dans {path}: {str(e)}")
        
        scan_recursive(directory_path)
        
    except Exception as e:
        error_list.append(f"Erreur générale: {str(e)}")
    
    return file_list, error_list

def analyze_file_content(file_path, verbose_errors=False):
    """Analyse universelle du contenu des fichiers avec gestion d'erreur améliorée"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return None
            
        file_info = {
            'name': file_path.name,
            'extension': file_path.suffix.lower(),
            'size': file_path.stat().st_size,
            'modified': datetime.fromtimestamp(file_path.stat().st_mtime),
            'keywords': [],
            'file_type': 'unknown'
        }
        
        # Analyse basée sur le nom
        name_lower = file_path.name.lower()
        
        # Détection de mots-clés dans le nom
        keywords_map = {
            'cv': ['cv', 'curriculum', 'resume'],
            'cover_letter': ['motivation', 'cover', 'lettre'],
            'invoice': ['facture', 'invoice', 'bill'],
            'contract': ['contrat', 'contract', 'agreement'],
            'report': ['rapport', 'report', 'summary'],
            'image': ['photo', 'image', 'picture'],
            'document': ['doc', 'document', 'text']
        }
        
        for category, keywords in keywords_map.items():
            for keyword in keywords:
                if keyword in name_lower:
                    file_info['keywords'].append(keyword)
                    file_info['file_type'] = category
                    break
        
        # Analyse du contenu pour certains types de fichiers
        try:
            if file_info['extension'] == '.txt':
                # Essayer différents encodages pour les fichiers texte
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
                    try:
                        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                            text = f.read(1000)  # Limiter à 1000 caractères
                        
                        text_lower = text.lower()
                        for category, keywords in keywords_map.items():
                            for keyword in keywords:
                                if keyword in text_lower:
                                    file_info['keywords'].append(keyword)
                                    if file_info['file_type'] == 'unknown':
                                        file_info['file_type'] = category
                                    break
                        break
                    except:
                        continue
            
            elif file_info['extension'] == '.csv':
                try:
                    # Essayer de lire le CSV avec pandas
                    df = pd.read_csv(file_path, nrows=5, encoding='utf-8', on_bad_lines='skip')
                    text = ' '.join(str(col) for col in df.columns)
                    
                    text_lower = text.lower()
                    for category, keywords in keywords_map.items():
                        for keyword in keywords:
                            if keyword in text_lower:
                                file_info['keywords'].append(keyword)
                                if file_info['file_type'] == 'unknown':
                                    file_info['file_type'] = category
                                break
                except:
                    pass
                    
        except Exception as e:
            if verbose_errors:
                st.warning(f"Erreur lors de l'analyse de contenu pour {file_path}: {str(e)}")
        
        # Classification automatique par extension
        extension_types = {
            'document': ['.pdf', '.docx', '.doc', '.txt', '.rtf', '.odt'],
            'spreadsheet': ['.xlsx', '.xls', '.csv', '.ods'],
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.webp'],
            'video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'],
            'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'],
            'archive': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'],
            'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.php', '.ipynb'],
            'presentation': ['.pptx', '.ppt', '.odp'],
            'data': ['.json', '.sql', '.db', '.parquet'],
            'executable': ['.exe', '.msi', '.deb', '.dmg', '.iso']
        }
        
        # Assigner le type basé sur l'extension
        for type_name, extensions in extension_types.items():
            if file_info['extension'] in extensions:
                if file_info['file_type'] == 'unknown':  # Ne pas écraser si déjà détecté
                    file_info['file_type'] = type_name
                break
        
        return file_info
        
    except Exception as e:
        if verbose_errors:
            st.error(f"Erreur lors de l'analyse de {file_path}: {str(e)}")
        return None

def analyze_directory_for_smart_classification(directory):
    """Analyse le dossier et propose une classification intelligente"""
    files, errors = scan_directory_universal(directory, max_depth=2)
    
    if not files:
        return None, errors
    
    # Analyser tous les fichiers
    file_analyses = []
    for file_path in files[:100]:  # Limiter à 100 fichiers pour la performance
        analysis = analyze_file_content(file_path)
        if analysis:
            file_analyses.append(analysis)
    
    if not file_analyses:
        return None, ["Aucun fichier analysable trouvé"]
    
    # Statistiques générales
    total_files = len(file_analyses)
    
    # Compter les types de fichiers
    file_types = Counter(analysis['file_type'] for analysis in file_analyses)
    extensions = Counter(analysis['extension'] for analysis in file_analyses)
    
    # Analyser les tailles
    sizes = [analysis['size'] for analysis in file_analyses]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    
    # Analyser les dates de modification
    dates = [analysis['modified'] for analysis in file_analyses]
    recent_threshold = datetime.now().replace(day=1)  # Premier jour du mois actuel
    recent_files = sum(1 for date in dates if date >= recent_threshold)
    
    # Générer des suggestions de classification
    suggestions = []
    
    # Suggestion 1: Par type de fichier (si un type domine)
    if file_types:
        most_common_type, count = file_types.most_common(1)[0]
        if count > total_files * 0.3:  # Si plus de 30% des fichiers sont du même type
            suggestions.append({
                'name': f'Organiser par type de fichier',
                'description': f'Séparer les {most_common_type} ({count} fichiers) du reste',
                'filter': {'file_type': most_common_type},
                'target_folder': f'Fichiers_{most_common_type}',
                'expected_files': count
            })
    
    # Suggestion 2: Par extension (si une extension domine)
    if extensions:
        most_common_ext, count = extensions.most_common(1)[0]
        if count > total_files * 0.25:  # Si plus de 25% des fichiers ont la même extension
            suggestions.append({
                'name': f'Organiser par extension',
                'description': f'Regrouper tous les fichiers {most_common_ext} ({count} fichiers)',
                'filter': {'extension': most_common_ext},
                'target_folder': f'Fichiers_{most_common_ext.replace(".", "")}',
                'expected_files': count
            })
    
    # Suggestion 3: Par taille (gros fichiers)
    large_files = sum(1 for size in sizes if size > 10 * 1024 * 1024)  # Plus de 10MB
    if large_files > 5:
        suggestions.append({
            'name': 'Séparer les gros fichiers',
            'description': f'Isoler les {large_files} fichiers volumineux (>10MB)',
            'filter': {'min_size': 10 * 1024 * 1024},
            'target_folder': 'Gros_fichiers',
            'expected_files': large_files
        })
    
    # Suggestion 4: Fichiers récents
    if recent_files > 10:
        suggestions.append({
            'name': 'Fichiers récents',
            'description': f'Regrouper les {recent_files} fichiers modifiés ce mois',
            'filter': {'recent': True},
            'target_folder': 'Fichiers_recents',
            'expected_files': recent_files
        })
    
    # Suggestion 5: Images si nombreuses
    image_count = file_types.get('image', 0)
    if image_count > 20:
        suggestions.append({
            'name': 'Organiser les images',
            'description': f'Regrouper toutes les {image_count} images',
            'filter': {'file_type': 'image'},
            'target_folder': 'Images',
            'expected_files': image_count
        })
    
    # Suggestion 6: Documents si nombreux
    doc_count = file_types.get('document', 0)
    if doc_count > 15:
        suggestions.append({
            'name': 'Organiser les documents',
            'description': f'Regrouper tous les {doc_count} documents',
            'filter': {'file_type': 'document'},
            'target_folder': 'Documents',
            'expected_files': doc_count
        })
    
    # Suggestion par défaut si aucune suggestion pertinente
    if not suggestions:
        suggestions.append({
            'name': 'Organisation générale',
            'description': 'Créer des dossiers par type de fichier',
            'filter': {},
            'target_folder': 'Organisation_generale',
            'expected_files': total_files
        })
    
    analysis_summary = {
        'total_files': total_files,
        'file_types': dict(file_types),
        'extensions': dict(extensions),
        'avg_size': avg_size,
        'recent_files': recent_files,
        'suggestions': suggestions[:5]  # Limiter à 5 suggestions max
    }
    
    return analysis_summary, errors

def organize_files_universal(source_dir, target_dir, filters, simulation=True):
    """Organisation universelle avec simulation"""
    results = {
        'moved': [],
        'errors': [],
        'skipped': [],
        'total_files': 0,
        'matched_files': 0
    }
    
    # Vérifier que le dossier source existe
    if not os.path.exists(source_dir):
        results['errors'].append(f"Le dossier source {source_dir} n'existe pas")
        return results
    
    files, scan_errors = scan_directory_universal(source_dir)
    results['errors'].extend(scan_errors)
    results['total_files'] = len(files)
    
    for file_path in files:
        file_info = analyze_file_content(file_path, verbose_errors=False)
        if not file_info:
            results['errors'].append(f"Impossible d'analyser: {file_path}")
            continue
        
        # Vérifier les filtres
        matches = True
        
        # Filtre par mot-clé
        if filters.get('keyword'):
            keyword_match = False
            keyword_lower = filters['keyword'].lower()
            
            # Chercher dans le nom du fichier
            if keyword_lower in file_info['name'].lower():
                keyword_match = True
            
            # Chercher dans les mots-clés détectés
            if any(keyword_lower in keyword.lower() for keyword in file_info['keywords']):
                keyword_match = True
            
            if not keyword_match:
                matches = False
        
        # Filtre par extension
        if filters.get('extension'):
            if isinstance(filters['extension'], list):
                if file_info['extension'] not in filters['extension']:
                    matches = False
            else:
                if file_info['extension'] != filters['extension'].lower():
                    matches = False
        
        # Filtre par type de fichier
        if filters.get('file_type') and file_info['file_type'] != filters['file_type']:
            matches = False
        
        # Filtre par taille
        if filters.get('min_size') and file_info['size'] < filters['min_size']:
            matches = False
        
        if filters.get('max_size') and file_info['size'] > filters['max_size']:
            matches = False
        
        # Filtre par date récente
        if filters.get('recent'):
            recent_threshold = datetime.now().replace(day=1)
            if file_info['modified'] < recent_threshold:
                matches = False
        
        if matches:
            results['matched_files'] += 1
            
            if not simulation:
                try:
                    # Créer le dossier de destination
                    target_path = Path(target_dir)
                    target_path.mkdir(parents=True, exist_ok=True)
                    
                    # Créer un nom unique si le fichier existe déjà
                    source_file = Path(file_path)
                    target_file = target_path / source_file.name
                    
                    counter = 1
                    while target_file.exists():
                        name_parts = source_file.stem, counter, source_file.suffix
                        target_file = target_path / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                        counter += 1
                    
                    # Déplacer le fichier
                    shutil.move(str(source_file), str(target_file))
                    results['moved'].append({
                        'source': file_path,
                        'target': str(target_file),
                        'info': file_info
                    })
                    
                except Exception as e:
                    results['errors'].append(f"Erreur lors du déplacement de {file_path}: {str(e)}")
            else:
                results['moved'].append({
                    'source': file_path,
                    'target': os.path.join(target_dir, file_info['name']),
                    'info': file_info
                })
        else:
            results['skipped'].append(file_path)
    
    return results

def main():
    st.set_page_config(
        page_title="AutoRanger",
        page_icon="📁",
        layout="wide"
    )
    
    st.title("AutoRanger - Organisez vos fichiers")
    
    # Initialiser les variables de session
    if 'notification' not in st.session_state:
        st.session_state.notification = None
    if 'source_dir' not in st.session_state:
        system_paths = get_system_paths()
        st.session_state.source_dir = system_paths["documents"]
    if 'smart_analysis' not in st.session_state:
        st.session_state.smart_analysis = None
    if 'selected_suggestion' not in st.session_state:
        st.session_state.selected_suggestion = None
    
    # Afficher la notification si elle existe
    if st.session_state.notification:
        if st.session_state.notification['type'] == 'success':
            st.success(st.session_state.notification['message'])
        elif st.session_state.notification['type'] == 'error':
            st.error(st.session_state.notification['message'])
        else:
            st.warning(st.session_state.notification['message'])
        # Effacer la notification après affichage
        st.session_state.notification = None
    
    # Sidebar pour le profil utilisateur
    with st.sidebar:
        st.header("Profil utilisateur")
        user_type = st.selectbox(
            "Type d'utilisateur",
            ["Débutant", "Intermédiaire", "Avancé", "Développeur"]
        )
        
        st.header("Raccourcis système")
        system_paths = get_system_paths()
        
        if st.button("📁 Documents"):
            st.session_state.source_dir = system_paths["documents"]
            st.session_state.smart_analysis = None
            st.session_state.selected_suggestion = None
            st.session_state.notification = {
                'type': 'success',
                'message': f"Dossier source défini sur: {system_paths['documents']}"
            }
            st.rerun()
            
        if st.button("⬇️ Téléchargements"):
            st.session_state.source_dir = system_paths["downloads"]
            st.session_state.smart_analysis = None
            st.session_state.selected_suggestion = None
            st.session_state.notification = {
                'type': 'success',
                'message': f"Dossier source défini sur: {system_paths['downloads']}"
            }
            st.rerun()
            
        if st.button("🖥️ Bureau"):
            st.session_state.source_dir = system_paths["desktop"]
            st.session_state.smart_analysis = None
            st.session_state.selected_suggestion = None
            st.session_state.notification = {
                'type': 'success',
                'message': f"Dossier source défini sur: {system_paths['desktop']}"
            }
            st.rerun()
    
    # Section d'analyse intelligente
    st.header("Analyse Intelligente")
    
    source_dir = st.text_input(
        "Dossier à analyser",
        value=st.session_state.get('source_dir', system_paths["documents"]),
        help="Chemin vers le dossier contenant les fichiers à organiser"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Analyser le dossier", type="primary"):
            if not source_dir:
                st.error("Veuillez spécifier un dossier à analyser")
            elif not os.path.exists(source_dir):
                st.error(f"Le dossier '{source_dir}' n'existe pas")
            else:
                with st.spinner("Analyse intelligente en cours..."):
                    try:
                        analysis, errors = analyze_directory_for_smart_classification(source_dir)
                        if analysis:
                            st.session_state.smart_analysis = analysis
                            st.session_state.source_dir = source_dir
                            st.success(f"Analyse terminée ! {analysis['total_files']} fichiers analysés")
                        else:
                            st.error("Impossible d'analyser le dossier")
                            if errors:
                                for error in errors:
                                    st.error(error)
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse: {str(e)}")
    
    with col2:
        if st.button("🔄 Nouvelle analyse"):
            st.session_state.smart_analysis = None
            st.session_state.selected_suggestion = None
            st.success("Prêt pour une nouvelle analyse")
    
    # Affichage des résultats d'analyse
    if st.session_state.smart_analysis:
        analysis = st.session_state.smart_analysis
        
        st.header("Résumé de l'analyse")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Fichiers analysés", analysis['total_files'])
        with col2:
            st.metric("Types différents", len(analysis['file_types']))
        with col3:
            st.metric("Extensions", len(analysis['extensions']))
        with col4:
            st.metric("Fichiers récents", analysis['recent_files'])
        
        # Graphiques de répartition
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Répartition par type")
            if analysis['file_types']:
                df_types = pd.DataFrame(list(analysis['file_types'].items()), 
                                      columns=['Type', 'Nombre'])
                st.bar_chart(df_types.set_index('Type'))
        
        with col2:
            st.subheader("Extensions les plus courantes")
            if analysis['extensions']:
                top_extensions = dict(sorted(analysis['extensions'].items(), 
                                           key=lambda x: x[1], reverse=True)[:10])
                df_ext = pd.DataFrame(list(top_extensions.items()), 
                                    columns=['Extension', 'Nombre'])
                st.bar_chart(df_ext.set_index('Extension'))
        
        # Suggestions de classification
        st.header("💡 Suggestions de classification")
        
        if analysis['suggestions']:
            for i, suggestion in enumerate(analysis['suggestions']):
                with st.expander(f"Suggestion {i+1}: {suggestion['name']}", expanded=(i==0)):
                    st.write(f"**Description:** {suggestion['description']}")
                    st.write(f"**Dossier de destination:** `{suggestion['target_folder']}`")
                    st.write(f"**Fichiers concernés:** {suggestion['expected_files']}")
                    
                    if st.button(f"Choisir cette suggestion", key=f"select_{i}"):
                        st.session_state.selected_suggestion = suggestion
                        st.success(f"Suggestion sélectionnée: {suggestion['name']}")
                        st.rerun()
        
        # Interface de validation de la suggestion sélectionnée
        if st.session_state.selected_suggestion:
            st.header("✅ Validation de la classification")
            
            suggestion = st.session_state.selected_suggestion
            
            st.info(f"**Classification choisie:** {suggestion['name']}")
            st.write(f"**Description:** {suggestion['description']}")
            
            # Permettre la modification du dossier de destination
            target_dir = st.text_input(
                "Dossier de destination",
                value=os.path.join(source_dir, suggestion['target_folder']),
                help="Vous pouvez modifier le nom du dossier de destination"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔍 Prévisualiser cette classification", type="primary"):
                    with st.spinner("Prévisualisation en cours..."):
                        try:
                            results = organize_files_universal(source_dir, target_dir, 
                                                             suggestion['filter'], simulation=True)
                            st.session_state.preview_results = results
                            if results['matched_files'] > 0:
                                st.success(f"Prévisualisation: {results['matched_files']} fichiers seront déplacés")
                            else:
                                st.warning("Aucun fichier ne correspond aux critères")
                        except Exception as e:
                            st.error(f"Erreur: {str(e)}")
            
            with col2:
                if st.button("✅ Appliquer la classification", type="secondary"):
                    if 'preview_results' not in st.session_state:
                        st.error("Veuillez d'abord faire une prévisualisation")
                    else:
                        with st.spinner("Classification en cours..."):
                            try:
                                results = organize_files_universal(source_dir, target_dir, 
                                                                 suggestion['filter'], simulation=False)
                                st.session_state.final_results = results
                                if results['moved']:
                                    st.success(f"Classification terminée ! {len(results['moved'])} fichiers déplacés")
                                else:
                                    st.warning("Aucun fichier n'a été déplacé")
                            except Exception as e:
                                st.error(f"Erreur: {str(e)}")
            
            with col3:
                if st.button("❌ Annuler"):
                    st.session_state.selected_suggestion = None
                    st.success("Sélection annulée")
                    st.rerun()
    
    # Mode manuel (interface existante simplifiée)
    if not st.session_state.smart_analysis or st.session_state.selected_suggestion is None:
        st.header("Classification manuelle")
        
        if user_type == "Débutant":
            col1, col2 = st.columns(2)
            
            with col1:
                target_dir = st.text_input(
                    "Dossier de destination",
                    value=os.path.join(source_dir if source_dir else "", "Fichiers_Organisés"),
                    help="Dossier où seront déplacés les fichiers"
                )
            
            with col2:
                preset = st.selectbox(
                    "Que voulez-vous organiser ?",
                    [
                        "Tous les CV et lettres de motivation",
                        "Toutes les factures (PDF)",
                        "Tous les fichiers ZIP",
                        "Toutes les images",
                        "Tous les fichiers de code",
                        "Tous les fichiers de données",
                        "Personnalisé"
                    ]
                )
            
            filters = {}
            if preset == "Tous les CV et lettres de motivation":
                filters = {'file_type': 'cv'}
            elif preset == "Toutes les factures (PDF)":
                filters = {'file_type': 'invoice', 'extension': '.pdf'}
            elif preset == "Tous les fichiers ZIP":
                filters = {'extension': '.zip'}
            elif preset == "Toutes les images":
                filters = {'file_type': 'image'}
            elif preset == "Tous les fichiers de code":
                filters = {'file_type': 'code'}
            elif preset == "Tous les fichiers de données":
                filters = {'file_type': 'data'}
            elif preset == "Personnalisé":
                keyword = st.text_input("Mot-clé dans le nom")
                extension = st.text_input("Extension (avec point, ex: .pdf)")
                if keyword:
                    filters['keyword'] = keyword
                if extension:
                    filters['extension'] = extension.lower()
        
        else:  # Modes Intermédiaire, Avancé, Développeur
            col1, col2, col3 = st.columns(3)
            
            with col1:
                target_dir = st.text_input(
                    "Dossier destination",
                    value=os.path.join(source_dir if source_dir else "", "Organisé")
                )
                keyword = st.text_input("Mot-clé", help="Recherche dans le nom et le contenu")
                
            with col2:
                extension = st.text_input("Extension (avec point, ex: .pdf)")
                file_type = st.selectbox(
                    "Type de fichier",
                    ["", "cv", "cover_letter", "invoice", "contract", "report", "image", 
                     "code", "data", "document", "spreadsheet", "video", "audio", "archive"]
                )
                
            with col3:
                if user_type in ["Avancé", "Développeur"]:
                    min_size = st.number_input("Taille min (bytes)", min_value=0, value=0)
                    max_size = st.number_input("Taille max (bytes)", min_value=0, value=0)
            
            filters = {}
            if keyword:
                filters['keyword'] = keyword
            if extension:
                filters['extension'] = extension.lower()
            if file_type:
                filters['file_type'] = file_type
            if user_type in ["Avancé", "Développeur"]:
                if min_size > 0:
                    filters['min_size'] = min_size
                if max_size > 0:
                    filters['max_size'] = max_size
        
        # Boutons d'action pour le mode manuel
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Prévisualiser (Manuel)", type="primary"):
                if not source_dir:
                    st.error("Veuillez spécifier un dossier source")
                elif not os.path.exists(source_dir):
                    st.error(f"Le dossier source '{source_dir}' n'existe pas")
                else:
                    with st.spinner("Analyse en cours..."):
                        try:
                            results = organize_files_universal(source_dir, target_dir, filters, simulation=True)
                            st.session_state.preview_results = results
                            if results['matched_files'] > 0:
                                st.success(f"Prévisualisation terminée: {results['matched_files']} fichiers correspondants trouvés sur {results['total_files']} analysés")
                            else:
                                st.warning("Aucun fichier ne correspond aux critères spécifiés")
                        except Exception as e:
                            st.error(f"Erreur lors de la prévisualisation: {str(e)}")
        
        with col2:
            if st.button("✅ Organiser (Manuel)", type="secondary"):
                if not source_dir:
                    st.error("Veuillez spécifier un dossier source")
                elif not os.path.exists(source_dir):
                    st.error(f"Le dossier source '{source_dir}' n'existe pas")
                elif 'preview_results' not in st.session_state:
                    st.error("Veuillez d'abord faire une prévisualisation")
                else:
                    with st.spinner("Organisation en cours..."):
                        try:
                            results = organize_files_universal(source_dir, target_dir, filters, simulation=False)
                            st.session_state.final_results = results
                            if results['moved']:
                                st.success(f"Organisation terminée ! {len(results['moved'])} fichiers déplacés")
                            else:
                                st.warning("Aucun fichier n'a été déplacé")
                        except Exception as e:
                            st.error(f"Erreur lors de l'organisation: {str(e)}")
        
        with col3:
            if st.button("🔄 Réinitialiser"):
                for key in ['preview_results', 'final_results']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Réinitialisation effectuée")
    
    # Mise à jour du dossier source dans la session
    st.session_state.source_dir = source_dir
    
    # Affichage des résultats de prévisualisation
    if 'preview_results' in st.session_state:
        results = st.session_state.preview_results
        
        st.header("🔍 Prévisualisation")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fichiers analysés", results['total_files'])
        with col2:
            st.metric("Fichiers correspondants", results['matched_files'])
        with col3:
            st.metric("Erreurs", len(results['errors']))
        
        if results['moved']:
            st.subheader("Fichiers qui seront déplacés:")
            df = pd.DataFrame([
                {
                    'Nom': item['info']['name'],
                    'Taille (bytes)': item['info']['size'],
                    'Type': item['info']['file_type'],
                    'Extension': item['info']['extension'],
                    'Source': item['source'],
                    'Destination': item['target']
                }
                for item in results['moved']
            ])
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Aucun fichier ne correspond aux critères de recherche")
        
        if results['errors']:
            with st.expander("Erreurs rencontrées"):
                for error in results['errors']:
                    st.error(error)
    
    # Affichage des résultats finaux
    if 'final_results' in st.session_state:
        results = st.session_state.final_results
        
        st.header("✅ Résultats de l'organisation")
        
        if results['moved']:
            st.subheader("Fichiers déplacés:")
            df = pd.DataFrame([
                {
                    'Nom': item['info']['name'],
                    'Taille (bytes)': item['info']['size'],
                    'Type': item['info']['file_type'],
                    'Extension': item['info']['extension'],
                    'Source': item['source'],
                    'Destination': item['target']
                }
                for item in results['moved']
            ])
            st.dataframe(df, use_container_width=True)
            
            # Bouton pour télécharger le rapport
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Télécharger le rapport CSV",
                data=csv,
                file_name=f"rapport_organisation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        if results['errors']:
            with st.expander("Erreurs rencontrées"):
                for error in results['errors']:
                    st.error(error)
    
    # Section d'aide
    with st.expander(" Aide et conseils"):
        st.markdown("""
        ###  Analyse Intelligente
        - **Analyser le dossier** : Examine automatiquement vos fichiers et propose des classifications pertinentes
        - **Suggestions** : Basées sur les types de fichiers, extensions, tailles et dates de modification
        - **Validation** : Prévisualisez toujours avant d'appliquer une classification
        
        ###  Mode Manuel
        - **Débutant** : Interface simplifiée avec des presets courants
        - **Avancé** : Filtres par taille, date, et critères personnalisés
        - **Simulation** : Testez toujours avec "Prévisualiser" avant d'organiser
        
        ###  Conseils
        - Faites des sauvegardes avant d'organiser des fichiers importants
        - Utilisez l'analyse intelligente pour découvrir des patterns dans vos fichiers
        - Les filtres peuvent être combinés pour des organisations précises
        - Le rapport CSV vous permet de garder une trace des déplacements
        """)

if __name__ == "__main__":
    main()