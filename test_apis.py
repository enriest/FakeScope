#!/usr/bin/env python3
"""
Quick API Test Script for FakeScope
Tests OpenAI, Gemini, and Perplexity APIs
"""
import os
import sys

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed, using environment variables only")

def test_openai():
    """Test OpenAI API"""
    print("\n🤖 Testing OpenAI API...")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("   ❌ OPENAI_API_KEY not set")
        return False
    
    print(f"   ✅ API Key found: {api_key[:10]}...")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'OpenAI works!'"}],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        print(f"   ✅ Response: {result}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Incorrect API key" in error_msg:
            print(f"   ❌ Invalid API key - get a new one from https://platform.openai.com/api-keys")
        else:
            print(f"   ❌ Error: {error_msg[:100]}...")
        return False

def test_gemini():
    """Test Google Gemini API"""
    print("\n🤖 Testing Google Gemini API...")
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("   ❌ GEMINI_API_KEY not set")
        return False
    
    print(f"   ✅ API Key found: {api_key[:10]}...")
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        candidates = [
            os.getenv("FAKESCOPE_GEMINI_MODEL", "gemini-1.5-flash"),
            "gemini-1.5-pro",
            "gemini-1.0-pro",
        ]
        last_err = None
        for m in candidates:
            try:
                model = genai.GenerativeModel(m)
                response = model.generate_content("Say 'Gemini works!'")
                result = response.text
                print(f"   ✅ Model {m}: {result}")
                return True
            except Exception as e:
                last_err = e
                continue
        # Fallback: raw REST call (bypasses SDK issues on older Python)
        import requests, json
        for m in candidates:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": "Say 'Gemini works!'"}]}]
            }
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            if res.ok:
                data = res.json()
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    print(f"   ✅ REST {m}: {text}")
                    return True
                except Exception:
                    pass
            last_err = res.text
        print(f"   ❌ Error: {str(last_err)}")
        return False
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def test_perplexity():
    """Test Perplexity API"""
    print("\n🤖 Testing Perplexity API...")
    api_key = os.getenv("PERPLEXITY_API_KEY")
    
    if not api_key:
        print("   ❌ PERPLEXITY_API_KEY not set")
        return False
    
    if api_key == "your-perplexity-key-here":
        print("   ❌ Please replace placeholder with actual Perplexity API key")
        print("   Get your key from: https://www.perplexity.ai/settings/api")
        return False
    
    print(f"   ✅ API Key found: {api_key[:10]}...")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        
        candidates = [
            os.getenv("FAKESCOPE_PERPLEXITY_MODEL", "llama-3.1-sonar-large-128k-online"),
            "sonar-pro",
            "sonar",
            "sonar-reasoning",
            "sonar-small-online",
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-large-128k-chat",
            "llama-3.1-70b-instruct",
        ]
        last_err = None
        for m in candidates:
            try:
                response = client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": "Say 'Perplexity works!'"}],
                    max_tokens=10
                )
                result = response.choices[0].message.content
                print(f"   ✅ Model {m}: {result}")
                return True
            except Exception as e:
                last_err = e
                continue
        print(f"   ❌ Error: {str(last_err)}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("🔍 FakeScope API Test Suite")
    print("=" * 60)
    
    results = {
        "OpenAI": test_openai(),
        "Gemini": test_gemini(),
        "Perplexity": test_perplexity()
    }
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    for provider, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"   {provider:12} {status}")
    
    working_count = sum(results.values())
    print("\n" + "=" * 60)
    print(f"✨ {working_count}/3 providers working")
    
    if working_count == 0:
        print("\n⚠️  No APIs working! Check your .env file and API keys.")
        sys.exit(1)
    elif working_count < 3:
        print("\n⚠️  Some APIs not configured. You can still use working providers.")
    else:
        print("\n🎉 All APIs working! You're ready to use FakeScope.")
    
    print("\nTo run FakeScope with a specific provider:")
    if results["Gemini"]:
        print("  export FAKESCOPE_LLM_PROVIDER=gemini     # ✅ FREE tier!")
    if results["OpenAI"]:
        print("  export FAKESCOPE_LLM_PROVIDER=openai     # ✅ Reliable")
    if results["Perplexity"]:
        print("  export FAKESCOPE_LLM_PROVIDER=perplexity # ✅ Real-time search")
    
    print("\nThen run:")
    print("  python -m streamlit run src/app.py")
    print()

if __name__ == "__main__":
    main()
