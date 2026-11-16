# Tests for the web page's request checks (SRS 5.4). Run from the project root:
#     python -m pytest tests
# No test calls Gemini: the runner is replaced by one that stops at once. Each
# test starts its own server on a free port, bound to the loopback address.
import http.client
import threading
import urllib.parse

import pytest

import web


def stop_at_once(question, job):
    raise RuntimeError("stopped by the test")


@pytest.fixture
def server():
    srv = web.make_server(port=0, runner=stop_at_once, explain=lambda error: "stopped")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def request(srv, method, path, headers=None, body=None):
    port = srv.server_address[1]
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    sent = {"Host": f"127.0.0.1:{port}"}
    sent.update(headers or {})
    connection.request(method, path, body=body, headers=sent)
    response = connection.getresponse()
    response.read()
    connection.close()
    return response


def post_question(srv, origin, fetch_site="same-origin"):
    body = urllib.parse.urlencode({"question": "How does attention work in a transformer?"})
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Content-Length": str(len(body))}
    if origin is not None:
        headers["Origin"] = origin
    if fetch_site is not None:
        headers["Sec-Fetch-Site"] = fetch_site
    return request(srv, "POST", "/run", headers, body)


def test_5_4_page_policy_lets_the_browser_send_its_real_origin(server):
    # Under no-referrer a browser sends "Origin: null" on the page's own form,
    # which the origin check refuses, so the Research button never worked
    response = request(server, "GET", "/")
    assert response.status == 200
    assert response.getheader("Referrer-Policy") == "same-origin"


def test_5_4_the_pages_own_form_post_is_accepted(server):
    port = server.server_address[1]
    response = post_question(server, f"http://127.0.0.1:{port}")
    assert response.status == 303
    assert response.getheader("Location").startswith("/job/")


def test_5_4_a_post_from_another_site_or_an_opaque_origin_is_refused(server):
    assert post_question(server, "http://evil.example", fetch_site="cross-site").status == 403
    assert post_question(server, "null").status == 403
