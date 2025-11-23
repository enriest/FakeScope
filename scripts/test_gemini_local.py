#!/usr/bin/env python3
"""
Local test script for Gemini API integration.
Run this to verify your Gemini API key and model access work correctly.

Usage:
    python test_gemini_local.py
"""

import os
import sys
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_api():
    """Test Gemini API with the same configuration as the Space."""
    
    print("=" * 60)
    print("FakeScope Gemini API Local Test")
    print("=" * 60)
    
    # Check API key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ GEMINI_API_KEY not found in environment")
        print("   Set it in .env file or export it:")
        print("   export GEMINI_API_KEY='your-key-here'")
        return False
    
    print(f"✅ GEMINI_API_KEY found ({len(gemini_key)} chars)")
    print(f"   Prefix: {gemini_key[:10]}...")
    print()
    
    # Test models in the same order as the app
    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"]
    
    test_prompt = "Hello! Please respond with 'API connection successful' if you can read this message."
    
    for model_name in models_to_try:
        print(f"Testing model: {model_name}")
        print("-" * 60)
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent"
            
            payload = {
                "contents": [{
                    "parts": [{"text": test_prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 100
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }
            
            print(f"   Making request to: {url}")
            resp = requests.post(url, params={"key": gemini_key}, json=payload, timeout=20)
            
            print(f"   Status code: {resp.status_code}")
            
            if resp.ok:
                data = resp.json()
                print(f"   Response keys: {list(data.keys())}")
                
                candidates = data.get("candidates", [])
                print(f"   Candidates: {len(candidates)}")
                
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
                    result = "\n".join([t for t in texts if t]).strip()
                    
                    if result:
                        print(f"   ✅ SUCCESS! Response length: {len(result)} chars")
                        print(f"   Response preview: {result[:200]}...")
                        print()
                        return True
                    else:
                        print(f"   ⚠️  No text in response")
                else:
                    print(f"   ⚠️  No candidates returned")
                    if "promptFeedback" in data:
                        print(f"   Prompt feedback: {data['promptFeedback']}")
                    print(f"   Full response: {data}")
            else:
                error_data = resp.json() if resp.headers.get('content-type', '').startswith('application/json') else {}
                error_msg = error_data.get("error", {}).get("message", resp.text[:500])
                print(f"   ❌ HTTP {resp.status_code}")
                print(f"   Error: {error_msg}")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Exception: {type(e).__name__}: {str(e)}")
            print()
    
    print("=" * 60)
    print("❌ All models failed")
    print("=" * 60)
    return False


def test_gemini_sdk():
    """Test using the official google-generativeai SDK as fallback."""
    
    print("\n" + "=" * 60)
    print("Testing with Official Gemini SDK")
    print("=" * 60)
    
    try:
        import google.generativeai as genai
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            print("❌ GEMINI_API_KEY not found")
            return False
        
        genai.configure(api_key=gemini_key)
        
        # Try to list models
        print("Listing available models...")
        models = genai.list_models()
        available_models = []
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                model_name = m.name.replace('models/', '')
                available_models.append(model_name)
                print(f"  - {model_name}")
        
        print()
        
        if available_models:
            # Try first available model
            test_model = available_models[0]
            print(f"Testing with: {test_model}")
            
            model = genai.GenerativeModel(test_model)
            response = model.generate_content("Say 'SDK test successful' if you can read this.")
            
            print(f"✅ SDK Test Success!")
            print(f"Response: {response.text[:200]}")
            return True
        else:
            print("❌ No models available for content generation")
            return False
            
    except ImportError:
        print("⚠️  google-generativeai not installed")
        print("   Install with: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ SDK test failed: {type(e).__name__}: {str(e)}")
        return False


if __name__ == "__main__":
    print("\n🔍 This script tests the Gemini API configuration")
    print("   used by FakeScope in production.\n")
    
    # Test REST API (same as production)
    rest_success = test_gemini_api()
    
    # Also test SDK for comparison
    sdk_success = test_gemini_sdk()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"REST API Test: {'✅ PASSED' if rest_success else '❌ FAILED'}")
    print(f"SDK Test:      {'✅ PASSED' if sdk_success else '❌ FAILED'}")
    print("=" * 60)
    
    if rest_success:
        print("\n✅ Your Gemini API configuration is working!")
        print("   The issue is likely environment-specific on Hugging Face.")
        print("\n   Check HF Spaces settings:")
        print("   1. Go to your Space settings")
        print("   2. Verify GEMINI_API_KEY is set correctly")
        print("   3. Check the deployment logs for the new [GEMINI] debug output")
    else:
        print("\n❌ Gemini API not working locally either.")
        print("\n   Troubleshooting steps:")
        print("   1. Verify your API key at: https://aistudio.google.com/apikey")
        print("   2. Check if the key has the right permissions")
        print("   3. Ensure you're not hitting rate limits")
        print("   4. Try regenerating your API key")
    
    sys.exit(0 if (rest_success or sdk_success) else 1)
