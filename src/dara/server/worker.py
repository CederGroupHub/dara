import logging
import time
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from traceback import format_exc

import ray
from jobflow.managers.local import run_locally
from monty.serialization import MontyDecoder

from dara.server.utils import get_job_store, get_result_store, get_worker_store

logger = logging.getLogger("dara.server.worker")


def worker_process():
    """Start the Ray worker process."""
    logger.info("Starting worker process for job execution...")
    mark_running_jobs_as_fizzled()

    while True:
        for job_uuid in get_all_pending_jobs():
            logger.debug(f"Job {job_uuid} has started...")
            run_job(job_uuid)
        time.sleep(3)


def run_job(uuid):
    """Run a job remotely by its UUID."""
    with get_worker_store() as worker_store:
        # launch ray earlier. To make sure it is run in a "pernament" folder that will not be deleted.
        if not ray.is_initialized():
            ray.init(runtime_env={"working_dir": None})

        job = worker_store.query_one(criteria={"uuid": uuid})
        job["start_time"] = datetime.now(tz=timezone.utc)
        job["status"] = "RUNNING"
        worker_store.update(job)
        try:
            with TemporaryDirectory() as tmp_dir:
                result = run_locally(
                    MontyDecoder().process_decoded(job["job"]),
                    raise_immediately=True,
                    store=get_job_store(get_result_store()),
                    root_dir=tmp_dir,
                )
            job["status"] = "COMPLETED"
            job["end_time"] = datetime.now(tz=timezone.utc)
            job["result"] = result
            worker_store.update(job)
            return result
        except Exception:
            job["status"] = "FIZZLED"
            job["error"] = format_exc()
            job["end_time"] = datetime.now(tz=timezone.utc)
            worker_store.update(job)
        except KeyboardInterrupt:
            job["status"] = "FIZZLED"
            job["error"] = format_exc()
            worker_store.update(job)
            raise


def add_job_to_queue(job, user):
    """Add a job to the queue for remote execution."""
    with get_worker_store() as worker_store:
        number_of_jobs = worker_store.count()
        worker_store.update(
            {
                "uuid": job.uuid,
                "job": job.as_dict(),
                "status": "READY",
                "submitted_time": datetime.now(tz=timezone.utc),
                "index": number_of_jobs + 1,
                "user": user,
            }
        )
    return number_of_jobs + 1  # index of the job in the queue


def get_all_pending_jobs(sort_by_submitted_time=False):
    """Add all pending jobs to the queue for remote execution."""
    with get_worker_store() as worker_store:
        jobs = worker_store.query(
            criteria={"status": "READY"},
            sort={"submitted_time": 1} if sort_by_submitted_time else None,
        )
        return [job["uuid"] for job in jobs]


def mark_running_jobs_as_fizzled():
    """Mark all running jobs as fizzled."""
    with get_worker_store() as worker_store:
        running_jobs = worker_store.query(criteria={"status": "RUNNING"})
        for job in running_jobs:
            job["status"] = "FIZZLED"
            job["end_time"] = datetime.now(tz=timezone.utc)
            job["error"] = "Unexpected shutdown of the worker process."
            worker_store.update(job)
            logger.warning(
                f"Job {job['uuid']} marked as fizzled due to previous shutdown."
            )
