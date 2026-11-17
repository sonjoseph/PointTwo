"""Run the local classifier; DATABASE_URL is provided outside source control."""

import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "classifier.api:app",
        host="127.0.0.1",
        port=int(os.getenv("POINTTWO_PORT", "8794")),
    )
