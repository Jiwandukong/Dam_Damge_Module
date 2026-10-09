#!/usr/bin/env python3
"""Serve only generated viewer assets, with an unguessable access link."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from functools import partial
import hmac
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import secrets
from urllib.parse import parse_qs,urlsplit
from paths import WORK


class Handler(SimpleHTTPRequestHandler):
    def log_message(self,fmt,*args):
        # Never write the access token or cookies into request logs.
        print(self.client_address[0],self.command,urlsplit(self.path).path,file=sys.stderr,flush=True)

    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        super().end_headers()

    def authorized(self):
        cookies=SimpleCookie()
        try:cookies.load(self.headers.get('Cookie',''))
        except Exception:return False
        value=cookies.get('rov_viewer')
        return bool(value and hmac.compare_digest(value.value,self.server.token))

    def send_head(self):
        parsed=urlsplit(self.path);token=parse_qs(parsed.query).get('token',[''])[0]
        if parsed.path=='/' and token and hmac.compare_digest(token,self.server.token):
            self.send_response(302);self.send_header('Location','/')
            self.send_header('Set-Cookie',f'rov_viewer={self.server.token}; HttpOnly; SameSite=Strict; Path=/')
            self.send_header('Content-Length','0');self.end_headers();return None
        if not self.authorized():
            self.send_error(403,'Use the private viewer access link.');return None
        path=Path(self.translate_path(parsed.path)).resolve()
        root=Path(self.directory).resolve()
        if path!=root and root not in path.parents:
            self.send_error(403);return None
        return super().send_head()

    def list_directory(self,path):
        self.send_error(403);return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bind',default='127.0.0.1');p.add_argument('--port',type=int,default=8788)
    a=p.parse_args();folder=WORK/'web_viewer';dist=folder/'dist'
    if not (dist/'index.html').exists():p.error('Run build_viewer.py first')
    token_file=folder/'access_token'
    if not token_file.exists():
        fd=os.open(token_file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(secrets.token_urlsafe(32))
    token_file.chmod(0o600)
    token=token_file.read_text().strip()
    server=ThreadingHTTPServer((a.bind,a.port),partial(Handler,directory=str(dist)))
    server.token=token
    print(f'ROV viewer ready on port {a.port}; local asset root: {dist}',flush=True)
    address = '127.0.0.1' if a.bind == '0.0.0.0' else a.bind
    print(f'접속 링크: http://{address}:{a.port}/?token={token}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
