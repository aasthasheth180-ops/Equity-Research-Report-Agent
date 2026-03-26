# publisher.py
#
# Publishes a generated equity research report (markdown) to WordPress.com
# via the standard WordPress REST API using Application Passwords.
#
# Setup (one-time, 2 minutes):
#   1. Log in to your WordPress.com dashboard
#   2. Go to: Users → Profile → Application Passwords
#   3. Create a new application password named "equity-research-agent"
#   4. Copy the password (shown once) into your .env file as WP_APP_PASSWORD
#
# Public API:
#   publish_report(report_md, title, status="draft") -> dict
#   list_posts(n=10) -> list

import base64
import os
from typing import List, Optional

import markdown as md_lib
import requests


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _auth_header() -> dict:
    """Build Basic-auth header from env vars."""
    username = os.getenv("WP_USERNAME", "")
    app_pass = os.getenv("WP_APP_PASSWORD", "")
    if not username or not app_pass:
        raise EnvironmentError(
            "Set WP_USERNAME and WP_APP_PASSWORD in your .env file.\n"
            "Get the application password from: WP Admin → Users → Profile → Application Passwords"
        )
    token = base64.b64encode(f"{username}:{app_pass}".encode()).decode()
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}


def _api_url(path: str) -> str:
    site = os.getenv("WP_SITE_URL", "").rstrip("/")
    if not site:
        raise EnvironmentError("Set WP_SITE_URL in your .env file, e.g. https://yoursite.wordpress.com")
    return f"{site}/wp-json/wp/v2/{path}"


# ── Markdown → HTML ───────────────────────────────────────────────────────────

def _md_to_html(md_text: str) -> str:
    """Convert report markdown to clean HTML with table + code support."""
    return md_lib.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br", "toc"],
    )


# ── WordPress helpers ─────────────────────────────────────────────────────────

def _get_or_create_tag(tag_name: str) -> int:
    """Return tag ID, creating it if it doesn't exist."""
    resp = requests.get(
        _api_url("tags"),
        params={"search": tag_name},
        headers=_auth_header(),
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()
    if results:
        return results[0]["id"]
    create = requests.post(
        _api_url("tags"),
        json={"name": tag_name},
        headers=_auth_header(),
        timeout=15,
    )
    create.raise_for_status()
    return create.json()["id"]


# ── Public API ────────────────────────────────────────────────────────────────

def publish_report(
    report_md:  str,
    title:      str,
    status:     str = "draft",          # "draft" | "publish"
    tags:       Optional[List[str]] = None,
    excerpt:    str = "",
) -> dict:
    """
    Convert markdown report to HTML and post to WordPress.com.

    Args:
        report_md:  Full markdown string (output of run_report)
        title:      Post title, e.g. "Goldman Sachs (GS) — Equity Research | Buy | PT $959"
        status:     "draft" to review before publishing, "publish" to go live immediately
        tags:       Optional list of tag strings, e.g. ["GS", "Equity Research", "Buy"]
        excerpt:    Short summary shown in post listings (optional)

    Returns:
        dict with post_id, url, edit_url, status
    """
    html_content = _md_to_html(report_md)

    tag_ids = [_get_or_create_tag(t) for t in (tags or [])]

    payload = {
        "title":   title,
        "content": html_content,
        "status":  status,
        "excerpt": excerpt,
        "format":  "standard",
    }
    if tag_ids:
        payload["tags"] = tag_ids

    resp = requests.post(
        _api_url("posts"),
        json=payload,
        headers=_auth_header(),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    site = os.getenv("WP_SITE_URL", "").rstrip("/")
    return {
        "post_id":  data["id"],
        "url":      data["link"],
        "status":   data["status"],
        "edit_url": f"{site}/wp-admin/post.php?post={data['id']}&action=edit",
    }


def list_posts(n: int = 10) -> list:
    """Return the n most recent posts (id, title, status, date, url)."""
    resp = requests.get(
        _api_url("posts"),
        params={"per_page": n, "orderby": "date", "order": "desc"},
        headers=_auth_header(),
        timeout=15,
    )
    resp.raise_for_status()
    return [
        {
            "id":     p["id"],
            "title":  p["title"]["rendered"],
            "status": p["status"],
            "date":   p["date"],
            "url":    p["link"],
        }
        for p in resp.json()
    ]
