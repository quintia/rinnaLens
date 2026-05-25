import http.server
import pathlib
import sys

PORT = 5055


def main():
    static_dir = pathlib.Path(__file__).parent / 'static'

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(static_dir), **kwargs)

    print(f'Running on http://localhost:{PORT}/')
    with http.server.HTTPServer(('', PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nStopped.')
            sys.exit(0)


if __name__ == '__main__':
    main()
