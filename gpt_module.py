import os

# Prefer the new google.genai client; fall back to google.generativeai.
_USE_NEW = False
_client = None
_MODEL = "gemini-1.5-flash-latest"

API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    print("Gemini: GOOGLE_API_KEY not found. Explanations will use a local fallback.")

try:
    from google import genai as google_genai
    if API_KEY:
        _client = google_genai.Client(api_key=API_KEY)
        _USE_NEW = True
        print("Gemini: using google.genai Client SDK.")
except Exception:
    pass

if not _USE_NEW:
    try:
        import google.generativeai as genai_old
        if API_KEY:
            genai_old.configure(api_key=API_KEY)
            _client = genai_old
            print("Gemini: using google.generativeai SDK.")
    except Exception:
        pass


def generate_explanation(crack_percent: float, condition: str) -> str:
    """
    Return a 6–7 line, student-project style engineering explanation.
    Never crashes: if Gemini not available, returns a local template.
    """
    prompt = f"""
Act as a mining engineer writing a short inspection note.
Inputs:
- Estimated crack coverage (percentage of image area): {crack_percent:.2f}%
- Condition class: {condition}

Write 6–7 short lines covering:
1) What the percentage implies for rock mass condition,
2) Key limitations (single image, lighting, texture, no instrument data),
3) Immediate action (visual re-check and closer images if needed),
4) Monitoring advice (periodic capture, same distance/angle),
5) Safety note if condition is Moderate or Poor,
6) One practical follow-up (simple support, marking, or scheduling a survey).
Keep the language simple and professional. write in bullet syntax; write as compact sentences in simple english.
"""

    # No key / clients → fallback text
    if not API_KEY or _client is None:
        cond_tip = "Increase vigilance and plan follow-up survey." if condition != "Good" else "Area appears serviceable with routine checks."
        return (
            f"The image-based estimate indicates about {crack_percent:.2f}% crack coverage, "
            f"classified as {condition}. This suggests the surface shows crack features detectable at this scale. "
            "Please note this is a vision-only approximation; lighting, shadows, and rock texture can bias results. "
            "Re-check on site and, if important, capture closer images at the same angle for confirmation. "
            "Consider periodic monitoring to track changes over time. "
            f"For current status: {cond_tip} "
            "If crack density increases or block instability is suspected, involve an expert for a detailed assessment."
        )

    try:
        if _USE_NEW:
            resp = _client.responses.generate(model=_MODEL, input=prompt)
            text = ""
            if hasattr(resp, "output_text") and resp.output_text:
                text = resp.output_text
            elif hasattr(resp, "candidates") and resp.candidates:
                # very defensive extraction
                cand = resp.candidates[0]
                text = getattr(getattr(cand, "content", None), "parts", [{}])[0].get("text", "") if hasattr(cand, "content") else ""
            return (text or "").strip()
        else:
            model = _client.GenerativeModel(_MODEL)
            r = model.generate_content(prompt)
            return (getattr(r, "text", None) or "").strip()
    except Exception as e:
        return (
            f"The image-based estimate is {crack_percent:.2f}% (condition: {condition}). "
            "This is a quick visual indicator and may be affected by lighting and surface texture. "
            "Verify in person, capture additional photos at consistent distance/angle, and monitor periodically. "
            "If the area is critical or shows worsening cracks, schedule a focused inspection."
        )
