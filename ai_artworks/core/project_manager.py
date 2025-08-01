"""
Project Management System
Advanced project handling with version control, collaboration, and cloud sync
"""

import json
import shutil
import hashlib
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QObject, Signal, QThread, QTimer
import git
import numpy as np

import logging
logger = logging.getLogger(__name__)


@dataclass
class ProjectMetadata:
    """Project metadata"""
    id: str
    name: str
    description: str
    created_at: datetime
    modified_at: datetime
    author: str
    version: str
    tags: List[str]
    thumbnail: Optional[str] = None
    collaborators: List[str] = None
    is_public: bool = False
    
    
@dataclass
class ProjectAsset:
    """Project asset information"""
    id: str
    path: str
    type: str  # 'image', 'model', 'preset', 'plugin'
    size: int
    checksum: str
    metadata: Dict[str, Any]
    

@dataclass
class ProjectVersion:
    """Project version information"""
    version: str
    timestamp: datetime
    author: str
    message: str
    changes: List[str]
    parent_version: Optional[str] = None


class ProjectDatabase:
    """SQLite database for project management"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Projects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP,
                    modified_at TIMESTAMP,
                    author TEXT,
                    version TEXT,
                    tags TEXT,
                    thumbnail TEXT,
                    collaborators TEXT,
                    is_public INTEGER,
                    data TEXT
                )
            """)
            
            # Assets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    path TEXT,
                    type TEXT,
                    size INTEGER,
                    checksum TEXT,
                    metadata TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)
            
            # Versions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT,
                    version TEXT,
                    timestamp TIMESTAMP,
                    author TEXT,
                    message TEXT,
                    changes TEXT,
                    parent_version TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)
            
            # Usage statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS usage_stats (
                    project_id TEXT,
                    last_opened TIMESTAMP,
                    open_count INTEGER DEFAULT 0,
                    total_time_seconds INTEGER DEFAULT 0,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)
            
            conn.commit()
            
    def add_project(self, project: ProjectMetadata) -> bool:
        """Add new project to database"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO projects (
                            id, name, description, created_at, modified_at,
                            author, version, tags, thumbnail, collaborators,
                            is_public, data
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        project.id,
                        project.name,
                        project.description,
                        project.created_at,
                        project.modified_at,
                        project.author,
                        project.version,
                        json.dumps(project.tags),
                        project.thumbnail,
                        json.dumps(project.collaborators or []),
                        int(project.is_public),
                        json.dumps(asdict(project))
                    ))
                    conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to add project: {e}")
                return False
                
    def get_project(self, project_id: str) -> Optional[ProjectMetadata]:
        """Get project by ID"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT data FROM projects WHERE id = ?",
                        (project_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        data = json.loads(row[0])
                        data['created_at'] = datetime.fromisoformat(data['created_at'])
                        data['modified_at'] = datetime.fromisoformat(data['modified_at'])
                        return ProjectMetadata(**data)
            except Exception as e:
                logger.error(f"Failed to get project: {e}")
        return None
        
    def list_projects(self, limit: int = 100, offset: int = 0) -> List[ProjectMetadata]:
        """List all projects"""
        projects = []
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT data FROM projects
                        ORDER BY modified_at DESC
                        LIMIT ? OFFSET ?
                    """, (limit, offset))
                    
                    for row in cursor.fetchall():
                        data = json.loads(row[0])
                        data['created_at'] = datetime.fromisoformat(data['created_at'])
                        data['modified_at'] = datetime.fromisoformat(data['modified_at'])
                        projects.append(ProjectMetadata(**data))
            except Exception as e:
                logger.error(f"Failed to list projects: {e}")
        return projects
        
    def search_projects(self, query: str) -> List[ProjectMetadata]:
        """Search projects by name, description, or tags"""
        projects = []
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT data FROM projects
                        WHERE name LIKE ? OR description LIKE ? OR tags LIKE ?
                        ORDER BY modified_at DESC
                    """, (f"%{query}%", f"%{query}%", f"%{query}%"))
                    
                    for row in cursor.fetchall():
                        data = json.loads(row[0])
                        data['created_at'] = datetime.fromisoformat(data['created_at'])
                        data['modified_at'] = datetime.fromisoformat(data['modified_at'])
                        projects.append(ProjectMetadata(**data))
            except Exception as e:
                logger.error(f"Failed to search projects: {e}")
        return projects


class VersionControl:
    """Git-based version control for projects"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.repo = None
        self._init_repo()
        
    def _init_repo(self):
        """Initialize git repository"""
        try:
            if (self.project_path / ".git").exists():
                self.repo = git.Repo(self.project_path)
            else:
                self.repo = git.Repo.init(self.project_path)
                # Initial commit
                self.repo.index.add("*")
                self.repo.index.commit("Initial project creation")
        except Exception as e:
            logger.error(f"Failed to initialize git repo: {e}")
            
    def commit(self, message: str, author: str = "AI-ARTWORKS") -> Optional[str]:
        """Create a new commit"""
        try:
            # Stage all changes
            self.repo.index.add("*")
            
            # Check if there are changes
            if self.repo.is_dirty():
                commit = self.repo.index.commit(
                    message,
                    author=git.Actor(author, f"{author}@ai-artworks.local")
                )
                return commit.hexsha
        except Exception as e:
            logger.error(f"Failed to commit: {e}")
        return None
        
    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get commit history"""
        history = []
        try:
            for commit in self.repo.iter_commits(max_count=limit):
                history.append({
                    'sha': commit.hexsha,
                    'message': commit.message,
                    'author': commit.author.name,
                    'timestamp': datetime.fromtimestamp(commit.committed_date),
                    'files': list(commit.stats.files.keys())
                })
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
        return history
        
    def checkout(self, commit_sha: str) -> bool:
        """Checkout a specific commit"""
        try:
            self.repo.git.checkout(commit_sha)
            return True
        except Exception as e:
            logger.error(f"Failed to checkout: {e}")
            return False
            
    def create_branch(self, branch_name: str) -> bool:
        """Create a new branch"""
        try:
            self.repo.create_head(branch_name)
            return True
        except Exception as e:
            logger.error(f"Failed to create branch: {e}")
            return False
            
    def merge_branch(self, branch_name: str) -> bool:
        """Merge a branch into current"""
        try:
            self.repo.git.merge(branch_name)
            return True
        except Exception as e:
            logger.error(f"Failed to merge branch: {e}")
            return False


class CloudSync(QThread):
    """Cloud synchronization for projects"""
    
    # Signals
    sync_started = Signal(str)
    sync_progress = Signal(str, int)
    sync_completed = Signal(str)
    sync_failed = Signal(str, str)
    
    def __init__(self, cloud_provider: str = "local"):
        super().__init__()
        self.cloud_provider = cloud_provider
        self.sync_queue = []
        self.running = True
        
    def add_to_sync(self, project_id: str, project_path: Path):
        """Add project to sync queue"""
        self.sync_queue.append((project_id, project_path))
        
    def run(self):
        """Sync thread"""
        while self.running:
            if self.sync_queue:
                project_id, project_path = self.sync_queue.pop(0)
                self._sync_project(project_id, project_path)
            else:
                self.msleep(1000)  # Sleep for 1 second
                
    def _sync_project(self, project_id: str, project_path: Path):
        """Sync single project"""
        self.sync_started.emit(project_id)
        
        try:
            # Calculate total size
            total_size = sum(f.stat().st_size for f in project_path.rglob("*") if f.is_file())
            uploaded_size = 0
            
            # Upload files
            for file_path in project_path.rglob("*"):
                if file_path.is_file():
                    # Simulate upload (replace with actual cloud API)
                    file_size = file_path.stat().st_size
                    self._upload_file(file_path, project_id)
                    
                    uploaded_size += file_size
                    progress = int((uploaded_size / total_size) * 100)
                    self.sync_progress.emit(project_id, progress)
                    
            self.sync_completed.emit(project_id)
            
        except Exception as e:
            self.sync_failed.emit(project_id, str(e))
            
    def _upload_file(self, file_path: Path, project_id: str):
        """Upload single file to cloud"""
        # Placeholder for actual cloud upload
        # This would integrate with AWS S3, Google Cloud Storage, etc.
        self.msleep(10)  # Simulate upload time


class ProjectManager(QObject):
    """Main project management system"""
    
    # Signals
    project_created = Signal(str)
    project_opened = Signal(str)
    project_saved = Signal(str)
    project_closed = Signal(str)
    project_deleted = Signal(str)
    
    def __init__(self, workspace_path: Path):
        super().__init__()
        
        self.workspace_path = workspace_path
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Database
        self.db = ProjectDatabase(workspace_path / "projects.db")
        
        # Current project
        self.current_project: Optional[ProjectMetadata] = None
        self.current_project_path: Optional[Path] = None
        self.version_control: Optional[VersionControl] = None
        
        # Cloud sync
        self.cloud_sync = CloudSync()
        self.cloud_sync.start()
        
        # Auto-save timer
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save)
        self.auto_save_timer.start(60000)  # Auto-save every minute
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def create_project(self, name: str, description: str = "", author: str = "User") -> Optional[str]:
        """Create new project"""
        try:
            # Generate project ID
            project_id = hashlib.sha256(
                f"{name}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]
            
            # Create project directory
            project_path = self.workspace_path / project_id
            project_path.mkdir(exist_ok=True)
            
            # Create subdirectories
            (project_path / "images").mkdir(exist_ok=True)
            (project_path / "models").mkdir(exist_ok=True)
            (project_path / "presets").mkdir(exist_ok=True)
            (project_path / "plugins").mkdir(exist_ok=True)
            (project_path / "exports").mkdir(exist_ok=True)
            
            # Create metadata
            metadata = ProjectMetadata(
                id=project_id,
                name=name,
                description=description,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                author=author,
                version="1.0.0",
                tags=[],
                collaborators=[author]
            )
            
            # Save metadata
            metadata_path = project_path / "project.json"
            with open(metadata_path, 'w') as f:
                data = asdict(metadata)
                data['created_at'] = data['created_at'].isoformat()
                data['modified_at'] = data['modified_at'].isoformat()
                json.dump(data, f, indent=2)
                
            # Add to database
            self.db.add_project(metadata)
            
            # Initialize version control
            VersionControl(project_path)
            
            self.project_created.emit(project_id)
            return project_id
            
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return None
            
    def open_project(self, project_id: str) -> bool:
        """Open existing project"""
        try:
            # Get project metadata
            metadata = self.db.get_project(project_id)
            if not metadata:
                return False
                
            # Check project directory
            project_path = self.workspace_path / project_id
            if not project_path.exists():
                return False
                
            # Close current project
            if self.current_project:
                self.close_project()
                
            # Set current project
            self.current_project = metadata
            self.current_project_path = project_path
            self.version_control = VersionControl(project_path)
            
            # Update usage stats
            self._update_usage_stats(project_id)
            
            self.project_opened.emit(project_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to open project: {e}")
            return False
            
    def save_project(self, create_version: bool = True) -> bool:
        """Save current project"""
        if not self.current_project:
            return False
            
        try:
            # Update metadata
            self.current_project.modified_at = datetime.now()
            
            # Save metadata
            metadata_path = self.current_project_path / "project.json"
            with open(metadata_path, 'w') as f:
                data = asdict(self.current_project)
                data['created_at'] = data['created_at'].isoformat()
                data['modified_at'] = data['modified_at'].isoformat()
                json.dump(data, f, indent=2)
                
            # Update database
            self.db.add_project(self.current_project)
            
            # Create version
            if create_version and self.version_control:
                self.version_control.commit(
                    f"Save project - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    self.current_project.author
                )
                
            # Queue for cloud sync
            self.cloud_sync.add_to_sync(
                self.current_project.id,
                self.current_project_path
            )
            
            self.project_saved.emit(self.current_project.id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            return False
            
    def close_project(self) -> bool:
        """Close current project"""
        if not self.current_project:
            return False
            
        # Save before closing
        self.save_project()
        
        project_id = self.current_project.id
        self.current_project = None
        self.current_project_path = None
        self.version_control = None
        
        self.project_closed.emit(project_id)
        return True
        
    def delete_project(self, project_id: str) -> bool:
        """Delete project"""
        try:
            # Close if current
            if self.current_project and self.current_project.id == project_id:
                self.close_project()
                
            # Remove from filesystem
            project_path = self.workspace_path / project_id
            if project_path.exists():
                shutil.rmtree(project_path)
                
            # Remove from database
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
                cursor.execute("DELETE FROM assets WHERE project_id = ?", (project_id,))
                cursor.execute("DELETE FROM versions WHERE project_id = ?", (project_id,))
                cursor.execute("DELETE FROM usage_stats WHERE project_id = ?", (project_id,))
                conn.commit()
                
            self.project_deleted.emit(project_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete project: {e}")
            return False
            
    def export_project(self, output_path: Path, include_history: bool = False) -> bool:
        """Export project as archive"""
        if not self.current_project:
            return False
            
        try:
            # Create zip archive
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all project files
                for file_path in self.current_project_path.rglob("*"):
                    if file_path.is_file():
                        # Skip git files if not including history
                        if not include_history and ".git" in str(file_path):
                            continue
                            
                        arcname = file_path.relative_to(self.current_project_path)
                        zipf.write(file_path, arcname)
                        
            return True
            
        except Exception as e:
            logger.error(f"Failed to export project: {e}")
            return False
            
    def import_project(self, archive_path: Path) -> Optional[str]:
        """Import project from archive"""
        try:
            # Extract to temp directory
            temp_path = self.workspace_path / "temp_import"
            temp_path.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(temp_path)
                
            # Load metadata
            metadata_path = temp_path / "project.json"
            if not metadata_path.exists():
                shutil.rmtree(temp_path)
                return None
                
            with open(metadata_path) as f:
                data = json.load(f)
                
            # Generate new project ID
            old_id = data['id']
            new_id = hashlib.sha256(
                f"{data['name']}_import_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]
            
            # Update metadata
            data['id'] = new_id
            data['created_at'] = datetime.now().isoformat()
            data['modified_at'] = datetime.now().isoformat()
            
            # Move to project directory
            project_path = self.workspace_path / new_id
            shutil.move(temp_path, project_path)
            
            # Save updated metadata
            with open(project_path / "project.json", 'w') as f:
                json.dump(data, f, indent=2)
                
            # Add to database
            metadata = ProjectMetadata(**data)
            metadata.created_at = datetime.fromisoformat(data['created_at'])
            metadata.modified_at = datetime.fromisoformat(data['modified_at'])
            self.db.add_project(metadata)
            
            return new_id
            
        except Exception as e:
            logger.error(f"Failed to import project: {e}")
            return None
            
    def list_recent_projects(self, limit: int = 10) -> List[ProjectMetadata]:
        """Get recently used projects"""
        return self.db.list_projects(limit=limit)
        
    def search_projects(self, query: str) -> List[ProjectMetadata]:
        """Search projects"""
        return self.db.search_projects(query)
        
    def get_project_stats(self, project_id: str) -> Dict[str, Any]:
        """Get project statistics"""
        stats = {
            'total_files': 0,
            'total_size': 0,
            'image_count': 0,
            'model_count': 0,
            'version_count': 0
        }
        
        try:
            project_path = self.workspace_path / project_id
            if project_path.exists():
                # Count files and size
                for file_path in project_path.rglob("*"):
                    if file_path.is_file():
                        stats['total_files'] += 1
                        stats['total_size'] += file_path.stat().st_size
                        
                        # Count by type
                        if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                            stats['image_count'] += 1
                        elif file_path.suffix.lower() in ['.pth', '.onnx', '.h5']:
                            stats['model_count'] += 1
                            
                # Count versions
                vc = VersionControl(project_path)
                stats['version_count'] = len(vc.get_history())
                
        except Exception as e:
            logger.error(f"Failed to get project stats: {e}")
            
        return stats
        
    def _update_usage_stats(self, project_id: str):
        """Update project usage statistics"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO usage_stats (project_id, last_opened, open_count)
                    VALUES (?, ?, COALESCE((SELECT open_count FROM usage_stats WHERE project_id = ?), 0) + 1)
                """, (project_id, datetime.now(), project_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update usage stats: {e}")
            
    def auto_save(self):
        """Auto-save current project"""
        if self.current_project:
            self.save_project(create_version=False)
            
    def cleanup(self):
        """Cleanup resources"""
        self.auto_save_timer.stop()
        self.cloud_sync.running = False
        self.cloud_sync.wait()
        self.executor.shutdown(wait=True)