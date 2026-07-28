from __future__ import annotations
import argparse
import uvicorn

def main():
    p=argparse.ArgumentParser(); p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=8000)
    a=p.parse_args(); uvicorn.run('goalkeeper_highlights.webapp:app',host=a.host,port=a.port,reload=False)
if __name__=='__main__': main()
