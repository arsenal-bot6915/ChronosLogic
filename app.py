import os
import re
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import List, Tuple
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from io import BytesIO


APP_NAME = "ChronosLogic"
BASE_DIR = Path(__file__).resolve().parent
# Always load .env from the script directory, so Streamlit run from其他目录也能生效.
DOTENV_PATH = BASE_DIR / ".env"
DOTENV_LOADED = load_dotenv(dotenv_path=str(DOTENV_PATH), override=True)


DEFAULT_SYSTEM_PROMPT = (
    "你是 ChronosLogic：擅长从史料中抽取逻辑链条，识别逻辑缺口，并给出可补全的建议。"
    "输出必须使用 Markdown。"
    "请严格使用以下 4 个主题作为大标题（每个主题是二级标题，且以数字序号开头）："
    "## 1. 逻辑链条"
    "## 2. 关键前提与结论"
    "## 3. 逻辑缺口"
    "## 4. 补全建议（可操作）"
    "不要输出除这 4 个主题之外的顶级大标题。"
)


def demo_analyze_logic(text: str) -> str:
    """
    演示版：在你尚未接入 AI API 之前先让页面跑起来。
    下一步你会把这里替换为真实的 API 调用与解析。
    """
    return f"""## 分析结果（演示版）

### 你输入的史料
> {text[:500].strip()}{'...' if len(text) > 500 else ''}

### 我建议你让 AI 输出的内容结构
1. **逻辑链条**：把史料里发生的事情按“因果/条件/时间顺序”拆出来。
2. **关键前提与结论**：列出哪些句子是前提，哪些句子是结论或推断。
3. **逻辑缺口**：指出缺少哪些信息会导致推理无法闭合。
4. **补全建议（可操作）**：给出需要补充的史料/证据类型、可验证的推断方式。

### 下一步（接入 AI API）
把 `analyze_logic_with_api()` 里对你实际 API 的调用补上，并把返回结果渲染到这里。
"""


def build_report_markdown(source_text: str, model: str, result_md: str) -> str:
    return f"""# ChronosLogic 分析报告

## 基础信息
- 模型：{model}
- 应用：{APP_NAME}

## 原始史料
{source_text}

## AI 分析结果
{result_md}
"""


def markdown_to_pdf_bytes(report_markdown: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    # Use built-in CJK font so Chinese text can render.
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    c.setFont("STSong-Light", 11)

    page_width, page_height = A4
    x = 45
    y = page_height - 45
    max_width = page_width - 90
    line_height = 16

    def draw_wrapped_line(text_line: str, y_pos: float) -> float:
        remaining = text_line
        while remaining:
            # Basic width-aware wrapping.
            cut = len(remaining)
            while cut > 1 and c.stringWidth(remaining[:cut], "STSong-Light", 11) > max_width:
                cut -= 1
            segment = remaining[:cut]
            c.drawString(x, y_pos, segment)
            remaining = remaining[cut:]
            y_pos -= line_height
            if y_pos < 45:
                c.showPage()
                c.setFont("STSong-Light", 11)
                y_pos = page_height - 45
        return y_pos

    for raw_line in report_markdown.splitlines():
        y = draw_wrapped_line(raw_line or " ", y)
        if y < 45:
            c.showPage()
            c.setFont("STSong-Light", 11)
            y = page_height - 45

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def analyze_logic_with_api(
    text: str,
    model: str,
    temperature: float,
    api_key: str,
    system_prompt: str,
) -> str:
    """
    使用 DeepSeek API（OpenAI SDK 兼容模式）
    """
    if not api_key:
        return demo_analyze_logic(text)
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"请分析下面史料的逻辑，并给出补全建议：\n\n{text}",
            },
        ],
        stream=False,
    )
    return response.choices[0].message.content or "模型返回为空。"


def split_result_markdown(result_md: str) -> List[Tuple[str, str, str]]:
    """
    将模型输出按如下格式拆分：
    ## 1. XXX
    ...内容...
    ## 2. YYY
    ...内容...
    """
    pattern = re.compile(r"^##\s*([1-4])\.\s*(.+?)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(result_md or ""))
    if not matches:
        return []

    sections: List[Tuple[str, str, str]] = []
    for i, m in enumerate(matches):
        num = m.group(1)
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(result_md)
        body = result_md[start:end].strip("\n ")
        sections.append((num, title, body))
    return sections


def save_feedback_to_file(feedback_text: str) -> Path:
    root = Path(__file__).resolve().parent
    logs_dir = root / "feedback_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    filename = logs_dir / f"feedback_{datetime.now().strftime('%Y%m%d')}.txt"
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {feedback_text}\n"
    with filename.open("a", encoding="utf-8") as f:
        f.write(line)
    return filename


def maybe_send_feedback_email(feedback_text: str) -> tuple[bool, str]:
    """
    只有在设置了 SMTP 环境变量时才会发送。
    需要的环境变量：
    - FEEDBACK_SMTP_HOST
    - FEEDBACK_SMTP_PORT
    - FEEDBACK_SMTP_USER
    - FEEDBACK_SMTP_PASS
    - FEEDBACK_TO_EMAIL
    （可选）FEEDBACK_FROM_EMAIL
    """
    host = os.getenv("FEEDBACK_SMTP_HOST", "").strip()
    port = os.getenv("FEEDBACK_SMTP_PORT", "").strip()
    user = os.getenv("FEEDBACK_SMTP_USER", "").strip()
    # Gmail App Password 通常以“xxxx xxxx xxxx xxxx”形式展示。
    # 复制到环境变量时可能保留空格，SMTP 登录时可能导致认证失败，因此去掉空格。
    # Gmail App Password 常见“分组展示”，复制后可能包含空格/引号
    pw = os.getenv("FEEDBACK_SMTP_PASS", "").strip()
    pw = pw.strip('"').strip("'").replace(" ", "")
    to_email = os.getenv("FEEDBACK_TO_EMAIL", "").strip()
    from_email = os.getenv("FEEDBACK_FROM_EMAIL", "").strip() or user

    if not all([host, port, user, pw, to_email, from_email]):
        missing = []
        if not host:
            missing.append("FEEDBACK_SMTP_HOST")
        if not port:
            missing.append("FEEDBACK_SMTP_PORT")
        if not user:
            missing.append("FEEDBACK_SMTP_USER")
        if not pw:
            missing.append("FEEDBACK_SMTP_PASS")
        if not to_email:
            missing.append("FEEDBACK_TO_EMAIL")
        if not from_email:
            missing.append("FEEDBACK_FROM_EMAIL")
        missing_str = "、".join(missing) if missing else "SMTP 配置项"
        dotenv_path = BASE_DIR / ".env"
        extra = ""
        if not dotenv_path.exists():
            extra = "；同时未检测到 .env 文件（脚本目录下）。"
        else:
            extra = f"；已检测到 .env 文件（大小：{dotenv_path.stat().st_size} bytes）。"
        return False, f"未配置 SMTP 邮件发送环境变量（缺少：{missing_str}），已仅保存到本地。{extra}"

    msg = EmailMessage()
    msg["Subject"] = f"{APP_NAME} - 反馈建议"
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(feedback_text)

    try:
        smtp_port = int(port)
        with smtplib.SMTP(host, smtp_port, timeout=20) as server:
            server.ehlo()
            # Gmail 通常是 587 + STARTTLS
            context = ssl.create_default_context()
            server.starttls(context=context)
            server.ehlo()
            server.login(user, pw)
            server.send_message(msg)
        return True, "邮件发送成功。"
    except Exception as e:
        return False, f"邮件发送失败：{e}"


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="centered")

    st.title(APP_NAME)
    st.caption("把史料输入给 ChronosLogic，输出逻辑补全建议。")
    st.info("请把 DeepSeek API Key 写入项目目录下的 `.env` 文件：`DEEPSEEK_API_KEY=...`（避免写进代码）。")

    if not DOTENV_LOADED:
        st.error("读取失败：找不到或无法读取项目目录下的 `.env` 文件。")

    env_api_key_preview = (os.getenv("DEEPSEEK_API_KEY", "") or "").strip()
    if not env_api_key_preview:
        st.warning("提示：`.env` 中未检测到 `DEEPSEEK_API_KEY`。你可以在输入框粘贴 key，或先补全 `.env`。")
    else:
        st.caption(f"读取到 `.env` 中的 `DEEPSEEK_API_KEY`（长度：{len(env_api_key_preview)}，格式：{'sk-' if env_api_key_preview.startswith('sk-') else '非 sk-' }）。")

    st.sidebar.title("使用指南")
    st.sidebar.markdown(
        "1. 在左侧输入史料内容（越完整越好）。\n"
        "2. 选择模型 `deepseek-chat`，设置 `temperature`（0.2 通常够用）。\n"
        "3. 提供 API Key（输入框或环境变量 `DEEPSEEK_API_KEY`）。\n"
        "4. 点击 `开始分析`，查看 4 个主题折叠框结果。\n"
        "5. 可下载 `Markdown` 或 `PDF` 报告。\n"
        "6. 底部填写“反馈建议”，会保存到本地；若你配置 SMTP，也会发到邮箱。"
    )

    st.sidebar.divider()
    st.sidebar.subheader("反馈邮件设置（可选，建议用）")
    with st.sidebar.expander("填写 SMTP/邮箱信息", expanded=False):
        smtp_host_in = st.text_input(
            "SMTP Host",
            value=os.getenv("FEEDBACK_SMTP_HOST", "smtp.gmail.com"),
        )
        smtp_port_in = st.text_input(
            "SMTP Port",
            value=os.getenv("FEEDBACK_SMTP_PORT", "587"),
        )
        smtp_user_in = st.text_input(
            "SMTP User",
            value=os.getenv("FEEDBACK_SMTP_USER", ""),
        )
        smtp_pass_in = st.text_input(
            "SMTP Pass（Gmail App Password）",
            type="password",
            value="",
        )
        feedback_to_in = st.text_input(
            "收件人 To",
            value=os.getenv("FEEDBACK_TO_EMAIL", ""),
        )
        feedback_from_in = st.text_input(
            "发件人 From",
            value=os.getenv("FEEDBACK_FROM_EMAIL", ""),
        )

    with st.form("chronoslogic_form", clear_on_submit=False):
        st.subheader("输入史料")
        text = st.text_area("请输入要分析的史料内容：", height=300)
        api_key_input = st.text_input(
            "DeepSeek API Key（可选；留空则读取环境变量 DEEPSEEK_API_KEY）",
            type="password",
            value="",
        )

        st.subheader("分析参数（可选）")
        model = st.text_input("模型名", value="deepseek-chat")
        temperature = st.slider("temperature（越大越发散）", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
        system_prompt = st.text_area(
            "提示词模板（System Prompt，可直接修改）",
            value=DEFAULT_SYSTEM_PROMPT,
            height=130,
        )

        submitted = st.form_submit_button("开始分析")

    if submitted:
        if not text.strip():
            st.warning("请先输入史料内容。")
            return

        env_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        api_key_input = (api_key_input or "").strip()
        api_key = (api_key_input or env_api_key).strip()
        key_source = "输入框" if api_key_input else ("环境变量 DEEPSEEK_API_KEY" if env_api_key else "未提供")
        if not api_key:
            st.error("读取失败：`.env` 中未找到 `DEEPSEEK_API_KEY`，且你未在输入框填写。请补全 `.env` 或在输入框粘贴 key。")
            return
        key_format_hint = "sk-开头" if api_key.startswith("sk-") else "非 sk- 格式（或未知）"
        st.caption(
            f"正在使用的 API Key 来源：{key_source}；Key 长度：{len(api_key)}；格式：{key_format_hint}"
        )

        with st.spinner("正在分析逻辑中..."):
            try:
                result_md = analyze_logic_with_api(
                    text=text,
                    model=model,
                    temperature=temperature,
                    api_key=api_key,
                    system_prompt=system_prompt,
                )
            except Exception as e:
                st.error(f"调用 DeepSeek API 失败：{e}")
                return

        st.subheader("分析结果")
        sections = split_result_markdown(result_md)
        if sections:
            for num, title, body in sections:
                with st.expander(f"{num}. {title}", expanded=(num == "1")):
                    st.markdown(body, unsafe_allow_html=False)
        else:
            st.markdown(result_md, unsafe_allow_html=False)

        report_md = build_report_markdown(text, model, result_md)
        pdf_bytes = markdown_to_pdf_bytes(report_md)

        st.subheader("导出报告")
        st.download_button(
            label="下载 Markdown 报告（.md）",
            data=report_md.encode("utf-8"),
            file_name="chronoslogic_report.md",
            mime="text/markdown",
        )
        st.download_button(
            label="下载 PDF 报告（.pdf）",
            data=pdf_bytes,
            file_name="chronoslogic_report.pdf",
            mime="application/pdf",
        )

    st.markdown("---")
    st.subheader("反馈建议")
    feedback_text = st.text_area("请填写你的建议/问题：", height=110, placeholder="例如：输出可以再更结构化一些；补全建议希望更可验证……")
    submit_feedback = st.button("提交反馈")

    if submit_feedback:
        feedback_text = (feedback_text or "").strip()
        if not feedback_text:
            st.warning("反馈内容不能为空。")
            return

        # 若用户在侧边栏填写了 SMTP 信息，就直接写入本次进程环境变量。
        # 这样可以绕过 .env 被覆盖/加载失败的问题。
        ui_smtp_fields = {
            "FEEDBACK_SMTP_HOST": smtp_host_in,
            "FEEDBACK_SMTP_PORT": smtp_port_in,
            "FEEDBACK_SMTP_USER": smtp_user_in,
            "FEEDBACK_SMTP_PASS": smtp_pass_in,
            "FEEDBACK_TO_EMAIL": feedback_to_in,
            "FEEDBACK_FROM_EMAIL": feedback_from_in,
        }
        using_ui_smtp = any((v or "").strip() for v in ui_smtp_fields.values())
        if using_ui_smtp:
            for k, v in ui_smtp_fields.items():
                v_str = (v or "").strip()
                if v_str:
                    os.environ[k] = v_str.replace(" ", "")
        else:
            # 反馈邮件的 SMTP 配置也建议直接写在 `.env` 中；此处不再依赖系统环境变量。
            # 如果 `.env` 中缺少 SMTP 信息，将提示仅保存到本地。
            pass

        filepath = save_feedback_to_file(feedback_text)
        st.success(f"已保存到：{filepath}")

        ok, msg = maybe_send_feedback_email(feedback_text)
        if ok:
            st.info(msg)
        else:
            st.caption(msg)
            st.caption(
                "如需自动发邮件，请设置环境变量："
                "`FEEDBACK_SMTP_HOST/FEEDBACK_SMTP_PORT/FEEDBACK_SMTP_USER/FEEDBACK_SMTP_PASS/FEEDBACK_TO_EMAIL`。"
            )


if __name__ == "__main__":
    main()