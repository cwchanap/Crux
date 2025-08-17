"""
Storage adapter for Cloudflare KV/D1
Provides abstraction layer for job and file storage
"""
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import aiofiles
import httpx
from pathlib import Path

class StorageAdapter:
    """
    Storage adapter that can work with:
    - Local file system (development)
    - Cloudflare KV (production)
    - Cloudflare D1 (production)
    """
    
    def __init__(self, storage_type: str = "local", config: Dict[str, Any] = None):
        self.storage_type = storage_type
        self.config = config or {}
        
        if storage_type == "local":
            self.base_path = Path(self.config.get("base_path", "./storage"))
            self.base_path.mkdir(exist_ok=True)
            (self.base_path / "jobs").mkdir(exist_ok=True)
            (self.base_path / "midi").mkdir(exist_ok=True)
        elif storage_type == "cloudflare_kv":
            self.account_id = self.config.get("account_id")
            self.namespace_id = self.config.get("namespace_id")
            self.api_token = self.config.get("api_token")
            self.kv_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/storage/kv/namespaces/{self.namespace_id}"
    
    async def save_job(self, job_id: str, job_data: Dict[str, Any]) -> bool:
        """Save job data to storage"""
        if self.storage_type == "local":
            job_path = self.base_path / "jobs" / f"{job_id}.json"
            async with aiofiles.open(job_path, "w") as f:
                await f.write(json.dumps(job_data, indent=2))
            return True
            
        elif self.storage_type == "cloudflare_kv":
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.kv_url}/values/job_{job_id}",
                    headers={
                        "Authorization": f"Bearer {self.api_token}",
                        "Content-Type": "application/json"
                    },
                    json=job_data
                )
                return response.status_code == 200
        
        return False
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job data from storage"""
        if self.storage_type == "local":
            job_path = self.base_path / "jobs" / f"{job_id}.json"
            if job_path.exists():
                async with aiofiles.open(job_path, "r") as f:
                    content = await f.read()
                    return json.loads(content)
            return None
            
        elif self.storage_type == "cloudflare_kv":
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.kv_url}/values/job_{job_id}",
                    headers={"Authorization": f"Bearer {self.api_token}"}
                )
                if response.status_code == 200:
                    return response.json()
        
        return None
    
    async def list_jobs(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """List all jobs with pagination"""
        if self.storage_type == "local":
            jobs_dir = self.base_path / "jobs"
            job_files = sorted(jobs_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            
            jobs = []
            for job_file in job_files[offset:offset + limit]:
                async with aiofiles.open(job_file, "r") as f:
                    content = await f.read()
                    jobs.append(json.loads(content))
            
            return jobs
            
        elif self.storage_type == "cloudflare_kv":
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.kv_url}/keys",
                    headers={"Authorization": f"Bearer {self.api_token}"},
                    params={"prefix": "job_", "limit": limit}
                )
                if response.status_code == 200:
                    keys = response.json().get("result", [])
                    jobs = []
                    for key_info in keys:
                        job_data = await self.get_job(key_info["name"].replace("job_", ""))
                        if job_data:
                            jobs.append(job_data)
                    return jobs
        
        return []
    
    async def save_midi(self, job_id: str, midi_data: bytes) -> bool:
        """Save MIDI file data"""
        if self.storage_type == "local":
            midi_path = self.base_path / "midi" / f"{job_id}.mid"
            async with aiofiles.open(midi_path, "wb") as f:
                await f.write(midi_data)
            return True
            
        elif self.storage_type == "cloudflare_kv":
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.kv_url}/values/midi_{job_id}",
                    headers={
                        "Authorization": f"Bearer {self.api_token}",
                        "Content-Type": "application/octet-stream"
                    },
                    content=midi_data
                )
                return response.status_code == 200
        
        return False
    
    async def get_midi(self, job_id: str) -> Optional[bytes]:
        """Get MIDI file data"""
        if self.storage_type == "local":
            midi_path = self.base_path / "midi" / f"{job_id}.mid"
            if midi_path.exists():
                async with aiofiles.open(midi_path, "rb") as f:
                    return await f.read()
            return None
            
        elif self.storage_type == "cloudflare_kv":
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.kv_url}/values/midi_{job_id}",
                    headers={"Authorization": f"Bearer {self.api_token}"}
                )
                if response.status_code == 200:
                    return response.content
        
        return None
    
    async def update_job_status(
        self, 
        job_id: str, 
        status: str, 
        progress: int = None,
        error: str = None,
        result_url: str = None
    ) -> bool:
        """Update job status and progress"""
        job_data = await self.get_job(job_id)
        if not job_data:
            return False
        
        job_data["status"] = status
        job_data["updated_at"] = datetime.utcnow().isoformat()
        
        if progress is not None:
            job_data["progress"] = progress
        if error is not None:
            job_data["error"] = error
        if result_url is not None:
            job_data["result_url"] = result_url
        
        return await self.save_job(job_id, job_data)
