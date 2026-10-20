from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    for width,height in [(375,812),(390,844),(320,700),(768,1024)]:
        page = browser.new_page(viewport={"width":width,"height":height},device_scale_factor=2,is_mobile=False,has_touch=True)
        errors=[]
        page.on("pageerror",lambda e: errors.append(str(e)))
        page.goto("http://127.0.0.1:8765/", wait_until="domcontentloaded")
        assert page.locator("#terms .term").count()==2
        assert page.locator("#searchMode").count()==0
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"),f"overflow at {width}"
        for target in ("#calc","#add",".operator",".phrase"):
            box=page.locator(target).first.bounding_box()
            assert box and box["height"]>=48, (width,target,box)
        page.locator("#add").click()
        assert page.locator("#terms .term").count()==3
        page.locator(".remove").last.click()
        assert page.locator("#terms .term").count()==2
        page.locator(".operator").nth(1).click()
        assert page.locator(".operator").nth(1).inner_text()=="+"
        page.locator(".operator").nth(1).click()
        page.locator("[data-example=king]").click()
        assert page.locator("#terms .term").count()==3
        page.locator("#calc").click()
        page.wait_for_function("!document.querySelector('#calc').disabled",timeout=90000)
        output=page.locator("#results").inner_text()
        assert "королева" in output.lower(),output
        assert not errors,errors
        print(f"PASS {width}x{height}; results={output[:90]!r}")
        page.close()
    browser.close()
