from playwright.sync_api import sync_playwright, expect

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    for width,height in ((320,700),(375,812),(390,844),(768,1024)):
        page=browser.new_page(viewport={"width":width,"height":height},has_touch=True)
        page.add_init_script("localStorage.setItem('vp.lang','ru');localStorage.setItem('vp.model','ru-rusvectores');")
        errors=[]
        page.on("pageerror",lambda exc:errors.append(str(exc)))
        page.goto("http://127.0.0.1:58765/",wait_until="domcontentloaded")
        assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
        page.locator(".phrase").first.wait_for(timeout=15000)
        assert page.locator(".phrase").count()==2
        page.locator("#add").click()
        assert page.locator(".phrase").count()==3
        page.locator(".remove").last.click()
        assert page.locator(".phrase").count()==2
        page.locator("[data-example=king]").click()
        page.locator("#calc").click(timeout=300000)
        expect(page.locator("#calc")).to_be_enabled(timeout=60000)
        expect(page.locator("#results")).to_contain_text("королева",timeout=60000)
        assert not errors,errors
        print("PASS",width,height)
        page.close()
    browser.close()
