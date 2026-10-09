"""Check the served viewer in Firefox with real WebGL rendering and point picking."""

import csv
import json
import os
from pathlib import Path
import select
import subprocess
from urllib.request import urlopen

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait

HERE = Path(__file__).resolve().parent
QA = HERE.parents[1] / "unmodified_gpr/GPR_work/viewer_runtime/qa"
URL = "http://127.0.0.1:8771/"


def main():
    QA.mkdir(parents=True, exist_ok=True)
    review = json.loads(urlopen(URL + "anomalies.json").read())
    with (HERE.parent / "Output/Result/ANM_result.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(review["candidates"]) == len(rows) == 126
    assert review['calibration']['horizontal_z_locked']
    horizontal_lines = [r for r in review['lines'] if r['line_no'] <= 3]
    assert len(horizontal_lines) == 12
    assert all(r['start_gltf'][1] == r['end_gltf'][1] for r in horizontal_lines)
    guide = review['anchor_proposal']
    assert guide['start']['world_xyz'][2] == guide['end']['world_xyz'][2]
    applied=review['calibration']['selected_anchors']
    reference=next(r for r in review['lines'] if r['freq_mhz']==400 and r['line_no']==1)
    for role in ['start','end']:
        xyz=applied[role]['world_xyz']
        assert reference[role+'_gltf']==[xyz[0],xyz[2],-xyz[1]]
    assert all(r['grid_id'] and r['member_name'] for r in rows)
    for row, point in zip(rows, review["candidates"]):
        assert point["damage_id"] == row["damage_id"]
        xyz = [float(row["world_center_" + a + "_m"]) for a in "xyz"]
        assert point["world_xyz"] == xyz
        assert point["gltf_xyz"] == [xyz[0], xyz[2], -xyz[1]]
    for image in (HERE.parent / "Output/Overlay/ANM").glob("*/*.png"):
        with urlopen(URL + "overlays/" + image.relative_to(HERE.parent / "Output/Overlay/ANM").as_posix()) as response:
            assert response.status == 200
    log = (QA / "browser.log").open("w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LIBGL_ALWAYS_SOFTWARE="1",
               GALLIUM_DRIVER="llvmpipe", MOZ_DISABLE_CONTENT_SANDBOX="1")
    xvfb = subprocess.Popen(["Xvfb", "-displayfd", "1", "-screen", "0", "1700x1100x24", "-nolisten", "tcp"],
                            stdout=subprocess.PIPE, stderr=log, text=True, start_new_session=True)
    driver = None
    try:
        if not select.select([xvfb.stdout], [], [], 10)[0]:
            raise RuntimeError("Xvfb unavailable")
        env["DISPLAY"] = ":" + xvfb.stdout.readline().strip()
        options = webdriver.FirefoxOptions()
        options.binary_location = "/snap/firefox/current/usr/lib/firefox/firefox"
        options.set_preference("webgl.force-enabled", True)
        options.set_preference("webgl.disabled", False)
        options.set_preference("browser.shell.checkDefaultBrowser", False)
        options.set_preference("datareporting.policy.dataSubmissionEnabled", False)
        service = Service("/snap/firefox/current/usr/lib/firefox/geckodriver", env=env, log_output=log)
        driver = webdriver.Firefox(service=service, options=options)
        driver.set_window_size(1600, 1000)
        driver.get(URL)
        wait = WebDriverWait(driver, 60)
        wait.until(lambda d: d.execute_script("return !!(window.MODEL_PREVIEW_READY||window.MODEL_PREVIEW_ERROR)"))
        error = driver.execute_script("return window.MODEL_PREVIEW_ERROR||null")
        if error:
            raise RuntimeError(error)
        print("WebGL model ready", flush=True)
        stats = driver.execute_script("return window.GPR_REVIEW_STATS")
        assert stats["total"] == stats["visible"] == 126
        assert stats["scale"] == 1 and stats["line_count"] == 48
        assert stats['calibration_status']=='selected_anchor_fit'
        assert stats['mapped_grid_count']==126
        assert stats['horizontal_distance_scale']==review['calibration']['horizontal_distance_scale']
        initial_anchors=driver.execute_script('return window.GPR_ANCHOR_API.record()')
        assert initial_anchors['start']['world_xyz']==applied['start']['world_xyz']
        assert initial_anchors['end']['world_xyz']==applied['end']['world_xyz']
        assert len(driver.find_elements(By.CSS_SELECTOR, ".candidate-row")) == 126
        wait.until(lambda d: d.execute_script('return [...document.querySelectorAll(".detail img")].every(i=>i.complete&&i.naturalWidth>0)'))
        driver.save_screenshot(str(QA / "dam_overview.png"))
        driver.find_element(By.ID, "candidate-fit").click()
        driver.save_screenshot(str(QA / "gpr_range.png"))
        driver.find_element(By.ID,"legacy").click()
        driver.save_screenshot(str(QA / "unit_scale_comparison.png"))
        driver.find_element(By.ID,"legacy").click()
        counts = {}
        for freq, count in [(400, 41), (900, 42), (1600, 26), (2600, 17)]:
            driver.execute_script("window.GPR_REVIEW_API.filter(arguments[0])", freq)
            visible = driver.execute_script("return window.GPR_REVIEW_STATS.visible")
            assert visible == count, (freq, visible)
            assert driver.execute_script("return window.GPR_REVIEW_STATS.line_count") == 12
            counts[str(freq)] = visible
        driver.execute_script("window.GPR_REVIEW_API.filter('all',1)")
        assert driver.execute_script("return window.GPR_REVIEW_STATS.visible") == sum(c["line_no"] == 1 for c in review["candidates"])
        driver.execute_script("window.GPR_REVIEW_API.filter();window.GPR_REVIEW_API.select('G000001')")
        driver.find_element(By.ID, "candidate-close").click()
        driver.execute_script("window.GPR_REVIEW_STATS.selected=null")
        ActionChains(driver).move_to_element(driver.find_element(By.ID, "canvas")).click().perform()
        picked = driver.execute_script("return window.GPR_REVIEW_STATS.selected")
        assert picked == "G000001", picked
        wait.until(lambda d: d.execute_script('return [...document.querySelectorAll(".detail img")].every(i=>i.complete&&i.naturalWidth>0)'))
        driver.save_screenshot(str(QA / "selected_point.png"))
        detail = driver.find_element(By.ID, "candidate-detail").text
        for value in review["candidates"][0]["world_xyz"]:
            assert f"{value:.6f}" in detail
        for layer in ["grid", "member", "all"]:
            driver.execute_script('document.getElementById("layer").value=arguments[0];document.getElementById("layer").dispatchEvent(new Event("change"));', layer)
        for mode in ["regions", "texture"]:
            driver.execute_script('document.getElementById("mode").value=arguments[0];document.getElementById("mode").dispatchEvent(new Event("change"));', mode)
        for checkbox in ["xray", "offset", "land", "survey-lines", "legacy"]:
            for value in [False, True]:
                driver.execute_script('const e=document.getElementById(arguments[0]);e.checked=arguments[1];e.dispatchEvent(new Event("change"));', checkbox, value)
        driver.find_element(By.ID, "top").click()
        driver.find_element(By.ID, "front").click()
        gl_error = driver.execute_script('return document.getElementById("canvas").getContext("webgl2").getError()')
        assert gl_error == 0, gl_error
        search = driver.find_element(By.ID, "search")
        search.send_keys("NO_SUCH_POINT")
        assert driver.execute_script("return window.GPR_REVIEW_STATS.visible") == 0
        search.clear()
        search.send_keys("G000126")
        assert driver.execute_script("return window.GPR_REVIEW_STATS.visible") == 1
        driver.execute_script("window.GPR_REVIEW_API.filter();window.GPR_REVIEW_API.fit();window.scrollTo(0,0)")
        driver.find_element(By.ID,"anchor-fit").click()
        driver.save_screenshot(str(QA / "anchor_proposal.png"))
        driver.find_element(By.ID,"anchor-start").click()
        ActionChains(driver).move_to_element(driver.find_element(By.ID,"canvas")).click().perform()
        anchor_start=driver.execute_script("return window.GPR_ANCHOR_API.record().start")
        assert anchor_start and anchor_start['mesh']=='NOF_R_0157',driver.find_element(By.ID,'anchor-status').text
        driver.find_element(By.ID,"anchor-end").click()
        ActionChains(driver).move_to_element(driver.find_element(By.ID,"canvas")).click().perform()
        anchor_end=driver.execute_script("return window.GPR_ANCHOR_API.record().end")
        assert anchor_end and anchor_end['mesh'].startswith('NOF_R_'),driver.find_element(By.ID,'anchor-status').text
        assert anchor_end['world_xyz'][2] == anchor_start['world_xyz'][2]
        assert anchor_end['raw_world_xyz'][2] != anchor_end['world_xyz'][2]
        # A second projection of the point onto the same mesh cross-section
        # must keep its position. Changing Z alone would put it off the mesh.
        projected = driver.execute_script('return window.GPR_ANCHOR_API.horizontalSurfacePoint(arguments[0],arguments[1])',anchor_end,anchor_start['world_xyz'][2])
        assert max(abs(a-b) for a,b in zip(projected['world_xyz'],anchor_end['world_xyz'])) < 1e-8
        assert driver.find_element(By.ID,'anchor-json').get_attribute('value')
        driver.save_screenshot(str(QA / "anchor_picks.png"))
        driver.find_element(By.ID,"anchor-clear").click()
        driver.set_window_size(390, 900)
        assert driver.execute_script("return document.documentElement.scrollWidth<=window.innerWidth+1")
        driver.save_screenshot(str(QA / "mobile.png"))
        result = dict(passed=True, model=driver.execute_script("return window.MODEL_PREVIEW_STATS"),
                      candidate_count=126, overlay_count=48, frequency_counts=counts, csv_xyz_preserved=True,
                      metric_scale=stats["scale"],survey_line_count=48,legacy_comparison_rendered=True,
                      horizontal_endpoint_ratio=stats['horizontal_distance_scale'],mapped_grid_count=126,
                      supplied_reference_endpoints_applied=True,
                      anchor_start_mesh=anchor_start['mesh'],anchor_end_mesh=anchor_end['mesh'],
                      horizontal_line_count=12,max_horizontal_endpoint_delta_z_m=0,
                      anchor_endpoint_delta_z_m=anchor_end['world_xyz'][2]-anchor_start['world_xyz'][2],
                      framebuffer_picked=picked, webgl_error=gl_error, mobile_no_horizontal_overflow=True)
        (QA / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception:
        if driver:
            driver.save_screenshot(str(QA / "failure.png"))
        raise
    finally:
        if driver:
            driver.quit()
        xvfb.terminate()
        xvfb.wait(timeout=10)
        log.close()


if __name__ == "__main__":
    main()
