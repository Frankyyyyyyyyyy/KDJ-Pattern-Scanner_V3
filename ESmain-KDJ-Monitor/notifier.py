"""
邮件通知模块（Gmail SMTP / 兼容标准 SMTPS）

使用 Gmail "应用专用密码"，不要用账户主密码：
    https://myaccount.google.com/apppasswords

所有连接信息从 config.EMAIL 读取；config.EMAIL.enabled 为 False 时
send() 成为 no-op，便于本地开发/测试。
"""

from __future__ import annotations
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr
from typing import Optional, Mapping

from config import EMAIL, logger


def _render_html(subject: str,
                 fields: Mapping[str, str],
                 note: str = "",
                 strength_color: str = "#1d1d1f") -> str:
    rows = "".join(
        f'<tr><td style="padding:6px 12px;background:#f5f5f7;font-weight:600;'
        f'border-bottom:1px solid #e5e5ea;white-space:nowrap;">{k}</td>'
        f'<td style="padding:6px 12px;border-bottom:1px solid #e5e5ea;'
        f'font-family:Menlo,Consolas,monospace;">{v}</td></tr>'
        for k, v in fields.items()
    )
    note_html = (
        f'<pre style="color:#3a3a3c;font-size:12px;background:#fafafa;'
        f'border:1px solid #e5e5ea;border-radius:4px;padding:10px 12px;'
        f'margin-top:18px;white-space:pre-wrap;font-family:inherit;">{note}</pre>'
        if note else ""
    )
    return f"""\
<!doctype html>
<html><body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                   max-width:600px;margin:24px auto;color:#1d1d1f;">
  <h2 style="margin:0 0 12px 0;border-left:5px solid {strength_color};
             padding-left:12px;line-height:1.3;">{subject}</h2>
  <table style="border-collapse:collapse;width:100%;border:1px solid #e5e5ea;
                border-radius:6px;overflow:hidden;">
    {rows}
  </table>
  {note_html}
  <p style="color:#86868b;font-size:12px;margin-top:24px;">
    — ESmain KDJ Monitor （脚本自动发送，请勿回复）
  </p>
</body></html>"""


def send(subject: str,
         fields: Mapping[str, str],
         note: str = "",
         text_body: Optional[str] = None,
         strength_color: str = "#1d1d1f") -> bool:
    """
    发送一封邮件。返回 True=发送成功，False=未启用 / 失败。
    """
    if not EMAIL.enabled:
        logger.info("邮件未启用（缺少 SENDER/RECEIVER/PASSWORD），跳过")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((EMAIL.sender_name, EMAIL.sender))
    msg["To"] = EMAIL.receiver

    if text_body is None:
        text_body = "\n".join(f"{k}: {v}" for k, v in fields.items())
        if note:
            text_body += f"\n\n{note}"
    msg.set_content(text_body)
    msg.add_alternative(
        _render_html(subject, fields, note, strength_color),
        subtype="html",
    )

    try:
        ctx = ssl.create_default_context()
        if EMAIL.smtp_port == 465:
            with smtplib.SMTP_SSL(EMAIL.smtp_host, EMAIL.smtp_port,
                                  context=ctx, timeout=30) as s:
                s.login(EMAIL.sender, EMAIL.password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(EMAIL.smtp_host, EMAIL.smtp_port, timeout=30) as s:
                s.starttls(context=ctx)
                s.login(EMAIL.sender, EMAIL.password)
                s.send_message(msg)
        logger.info(f"邮件已发送 → {EMAIL.receiver}")
        return True
    except Exception as e:
        logger.error(f"邮件发送失败: {e}")
        return False
