document.addEventListener('DOMContentLoaded', () => {
  // Global State
  let currentUserSession = null;
  let currentMode = 'dual'; // 'dual', 'classify', 'segment'
  let selectedFile = null;
  let isRegistering = false;

  // DOM Elements - Auth & App Shell
  const appLoadingScreen = document.getElementById('appLoadingScreen');
  const authPage = document.getElementById('authPage');
  const appShell = document.getElementById('appShell');

  // DOM Elements - Auth Form
  const authForm = document.getElementById('authForm');
  const authFormTitle = document.getElementById('authFormTitle');
  const authFormSubtitle = document.getElementById('authFormSubtitle');
  const tabSignIn = document.getElementById('tabSignIn');
  const tabSignUp = document.getElementById('tabSignUp');
  const authErrorBox = document.getElementById('authErrorBox');
  const authEmailInput = document.getElementById('authEmail');
  const authPasswordInput = document.getElementById('authPassword');
  const authUsernameInput = document.getElementById('authUsername');
  const authRoleInput = document.getElementById('authRole');
  const usernameGroup = document.getElementById('usernameGroup');
  const roleGroup = document.getElementById('roleGroup');
  const authSubmitBtn = document.getElementById('authSubmitBtn');
  const authSubmitText = document.getElementById('authSubmitText');
  const fillDemoBtn = document.getElementById('fillDemoBtn');
  const togglePasswordBtn = document.getElementById('togglePasswordBtn');
  const togglePasswordIcon = document.getElementById('togglePasswordIcon');

  // DOM Elements - Navigation & Profile
  const authBtn = document.getElementById('authBtn');
  const navUserName = document.getElementById('navUserName');
  const navAvatarChip = document.getElementById('navAvatarChip');
  const userProfileModal = document.getElementById('userProfileModal');
  const closeProfileModal = document.getElementById('closeProfileModal');
  const profileUsername = document.getElementById('profileUsername');
  const profileRole = document.getElementById('profileRole');
  const profileEmail = document.getElementById('profileEmail');
  const profileAvatarLarge = document.getElementById('profileAvatarLarge');
  const logoutBtn = document.getElementById('logoutBtn');

  // DOM Elements - Workspace Controls & Results
  const modeTabs = document.querySelectorAll('.mode-tab');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const filePreview = document.getElementById('filePreview');
  const previewImg = document.getElementById('previewImg');
  const scanFilenameVal = document.getElementById('scanFilenameVal');
  const scanFilesizeVal = document.getElementById('scanFilesizeVal');
  const replaceFileBtn = document.getElementById('replaceFileBtn');
  const removeFileBtn = document.getElementById('removeFileBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const analyzeBtnText = document.getElementById('analyzeBtnText');

  const placeholderView = document.getElementById('placeholderView');
  const resultsSection = document.getElementById('resultsSection');
  const originalScanImg = document.getElementById('originalScanImg');
  const overlayScanImg = document.getElementById('overlayScanImg');
  const maskScanImg = document.getElementById('maskScanImg');
  const maskStageBox = document.getElementById('maskStageBox');
  const overlayStageBox = document.getElementById('overlayStageBox');
  const opacityBarWrapper = document.getElementById('opacityBarWrapper');

  const classBadge = document.getElementById('classBadge');
  const confidenceBadge = document.getElementById('confidenceBadge');
  const tumorAreaBadge = document.getElementById('tumorAreaBadge');
  const tumorPctBadge = document.getElementById('tumorPctBadge');
  const classMetricCard = document.getElementById('classMetricCard');
  const confMetricCard = document.getElementById('confMetricCard');
  const tumorPctCard = document.getElementById('tumorPctCard');
  const tumorAreaCard = document.getElementById('tumorAreaCard');

  const probBarsContainer = document.getElementById('probBarsContainer');
  const probSection = document.getElementById('probSection');
  const modelUsedFootnote = document.getElementById('modelUsedFootnote');
  const opacitySlider = document.getElementById('opacitySlider');
  const opacityValText = document.getElementById('opacityValText');

  const imageInspectModal = document.getElementById('imageInspectModal');
  const closeInspectModal = document.getElementById('closeInspectModal');
  const inspectModalImg = document.getElementById('inspectModalImg');
  const expandPreviewBtn = document.getElementById('expandPreviewBtn');

  // ==========================================
  // 1. AUTHENTICATION FLOW & ROUTE GUARDS
  // ==========================================
  async function checkSessionOnBoot() {
    try {
      const res = await fetch('/api/auth/session');
      const data = await res.json();

      if (data.logged_in && data.user) {
        currentUserSession = data.user;
        updateUserProfileUI(data.user);
        showAppShell();
        handleHashRouting();
      } else {
        currentUserSession = null;
        showAuthPage();
      }
    } catch (err) {
      console.error('Boot auth check error:', err);
      currentUserSession = null;
      showAuthPage();
    }
  }

  function showAppShell() {
    if (appLoadingScreen) appLoadingScreen.style.display = 'none';
    if (authPage) authPage.style.display = 'none';
    if (appShell) appShell.style.display = 'block';
  }

  function showAuthPage() {
    if (appLoadingScreen) appLoadingScreen.style.display = 'none';
    if (appShell) appShell.style.display = 'none';
    if (authPage) authPage.style.display = 'block';

    // Replace browser history state to prevent Back button revealing protected workspace
    if (window.location.hash && window.location.hash !== '#auth') {
      window.history.replaceState(null, '', '#auth');
    }
  }

  function updateUserProfileUI(user) {
    const name = user.username || 'Sapna';
    const role = user.role || 'Doctor / Radiologist';
    const email = user.email || 'sapna26@gmail.com';
    const initials = name.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase() || 'DS';

    if (navUserName) navUserName.textContent = `Dr. ${name}`;
    if (navAvatarChip) navAvatarChip.textContent = initials;
    if (profileUsername) profileUsername.textContent = `Dr. ${name}`;
    if (profileRole) profileRole.textContent = role;
    if (profileEmail) profileEmail.textContent = email;
    if (profileAvatarLarge) profileAvatarLarge.textContent = initials;
  }

  // Hash & View Routing Guard
  function handleHashRouting() {
    if (!currentUserSession) {
      showAuthPage();
      return;
    }

    const hash = window.location.hash.replace('#', '') || 'workspace';
    if (hash === 'analytics') {
      switchView('analytics');
    } else if (hash === 'history') {
      switchView('history');
    } else {
      switchView('workspace');
    }
  }

  window.addEventListener('popstate', () => {
    if (!currentUserSession) {
      showAuthPage();
    } else {
      handleHashRouting();
    }
  });

  window.switchView = function(viewName) {
    if (!currentUserSession) {
      showAuthPage();
      return;
    }

    const workspaceEl = document.getElementById('workspaceSection');
    const analyticsEl = document.getElementById('analyticsSection');
    const historyEl = document.getElementById('historySection');

    const navW = document.getElementById('navWorkspace');
    const navA = document.getElementById('navAnalytics');
    const navH = document.getElementById('navHistory');

    if (navW) navW.classList.remove('active');
    if (navA) navA.classList.remove('active');
    if (navH) navH.classList.remove('active');

    if (workspaceEl) workspaceEl.style.display = 'none';
    if (analyticsEl) analyticsEl.style.display = 'none';
    if (historyEl) historyEl.style.display = 'none';

    if (viewName === 'workspace') {
      if (navW) navW.classList.add('active');
      if (workspaceEl) workspaceEl.style.display = 'block';
      window.history.pushState(null, '', '#workspace');
    } else if (viewName === 'analytics') {
      if (navA) navA.classList.add('active');
      if (analyticsEl) analyticsEl.style.display = 'block';
      window.history.pushState(null, '', '#analytics');
      if (typeof window.loadAnalytics === 'function') window.loadAnalytics();
    } else if (viewName === 'history') {
      if (navH) navH.classList.add('active');
      if (historyEl) historyEl.style.display = 'block';
      window.history.pushState(null, '', '#history');
      if (typeof window.loadHistory === 'function') window.loadHistory();
    }
  };

  // Sign Out Execution
  async function performSignOut() {
    try {
      await fetch('/api/auth/logout', { method: 'POST' });
    } catch (err) {
      console.error('Logout error:', err);
    } finally {
      currentUserSession = null;
      if (userProfileModal) userProfileModal.classList.remove('open');
      showAuthPage();
    }
  }

  if (logoutBtn) logoutBtn.addEventListener('click', performSignOut);

  // ==========================================
  // 2. AUTHENTICATION FORM CONTROLS
  // ==========================================
  function switchAuthTab(register) {
    isRegistering = register;
    clearAuthError();

    if (register) {
      tabSignUp.classList.add('active');
      tabSignUp.setAttribute('aria-selected', 'true');
      tabSignIn.classList.remove('active');
      tabSignIn.setAttribute('aria-selected', 'false');

      if (authFormTitle) authFormTitle.textContent = 'Create an account';
      if (authFormSubtitle) authFormSubtitle.textContent = 'Register for diagnostic workspace access.';
      if (usernameGroup) usernameGroup.style.display = 'block';
      if (roleGroup) roleGroup.style.display = 'block';
      if (authSubmitText) authSubmitText.textContent = 'Create Account';
    } else {
      tabSignIn.classList.add('active');
      tabSignIn.setAttribute('aria-selected', 'true');
      tabSignUp.classList.remove('active');
      tabSignUp.setAttribute('aria-selected', 'false');

      if (authFormTitle) authFormTitle.textContent = 'Welcome back';
      if (authFormSubtitle) authFormSubtitle.textContent = 'Sign in to access your diagnostic workspace.';
      if (usernameGroup) usernameGroup.style.display = 'none';
      if (roleGroup) roleGroup.style.display = 'none';
      if (authSubmitText) authSubmitText.textContent = 'Sign In to NeuroScan';
    }
  }

  if (tabSignIn) tabSignIn.addEventListener('click', () => switchAuthTab(false));
  if (tabSignUp) tabSignUp.addEventListener('click', () => switchAuthTab(true));

  function showAuthError(msg) {
    if (authErrorBox) {
      authErrorBox.textContent = msg;
      authErrorBox.style.display = 'block';
    }
  }

  function clearAuthError() {
    if (authErrorBox) {
      authErrorBox.textContent = '';
      authErrorBox.style.display = 'none';
    }
  }

  if (togglePasswordBtn && authPasswordInput) {
    togglePasswordBtn.addEventListener('click', () => {
      const isPass = authPasswordInput.type === 'password';
      authPasswordInput.type = isPass ? 'text' : 'password';
      togglePasswordBtn.setAttribute('aria-label', isPass ? 'Hide password' : 'Show password');
      if (togglePasswordIcon) {
        togglePasswordIcon.className = isPass ? 'fas fa-eye-slash' : 'fas fa-eye';
      }
    });
  }

  if (fillDemoBtn) {
    fillDemoBtn.addEventListener('click', () => {
      switchAuthTab(false);
      if (authEmailInput) authEmailInput.value = 'sapna26@gmail.com';
      if (authPasswordInput) authPasswordInput.value = 'Sapna@123';
      clearAuthError();
    });
  }

  if (authForm) {
    authForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      clearAuthError();

      const email = authEmailInput ? authEmailInput.value.trim() : '';
      const password = authPasswordInput ? authPasswordInput.value : '';
      const username = authUsernameInput ? authUsernameInput.value.trim() : '';
      const role = authRoleInput ? authRoleInput.value : 'Doctor/Radiologist';

      if (!email || !password) {
        showAuthError('Please enter email address and password.');
        return;
      }
      if (isRegistering && !username) {
        showAuthError('Please enter your full name or username.');
        return;
      }

      if (authSubmitBtn) authSubmitBtn.disabled = true;
      if (authSubmitText) authSubmitText.textContent = isRegistering ? 'Creating account...' : 'Signing in...';

      const endpoint = isRegistering ? '/api/auth/register' : '/api/auth/login';
      const payload = isRegistering
        ? { username, email, password, role }
        : { email_or_username: email, password };

      try {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        if (data.success) {
          currentUserSession = data.user || { username: username || 'Sapna', email, role };
          updateUserProfileUI(currentUserSession);
          showAppShell();
          handleHashRouting();
        } else {
          showAuthError(data.message || 'Authentication failed. Please check credentials.');
        }
      } catch (err) {
        console.error('Auth submit error:', err);
        showAuthError('Unable to connect to authentication server. Please try again.');
      } finally {
        if (authSubmitBtn) authSubmitBtn.disabled = false;
        if (authSubmitText) authSubmitText.textContent = isRegistering ? 'Create Account' : 'Sign In to NeuroScan';
      }
    });
  }

  // Profile Modal Trigger
  if (authBtn) {
    authBtn.addEventListener('click', () => {
      if (userProfileModal) userProfileModal.classList.add('open');
    });
  }

  if (closeProfileModal) {
    closeProfileModal.addEventListener('click', () => {
      if (userProfileModal) userProfileModal.classList.remove('open');
    });
  }

  // ==========================================
  // 3. DIAGNOSTIC WORKSPACE & PIPELINE CONTROLS
  // ==========================================
  modeTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      modeTabs.forEach(t => {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
      });
      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');
      currentMode = tab.dataset.mode;
      updateAnalyzeBtnText();
    });
  });

  function updateAnalyzeBtnText() {
    if (!analyzeBtnText) return;
    if (currentMode === 'dual') {
      analyzeBtnText.textContent = 'Run Dual AI Analysis';
    } else if (currentMode === 'classify') {
      analyzeBtnText.textContent = 'Run VGG16 Classification';
    } else {
      analyzeBtnText.textContent = 'Run U-Net Segmentation';
    }
  }

  // Dropzone Interaction
  if (dropzone) {
    dropzone.addEventListener('click', (e) => {
      if (e.target.closest('input')) return;
      if (fileInput) fileInput.click();
    });

    ['dragenter', 'dragover'].forEach(evt => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(evt => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) handleFileSelected(files[0]);
    });
  }

  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) handleFileSelected(e.target.files[0]);
    });
  }

  const btnHelperText = document.getElementById('btnHelperText');

  function handleFileSelected(file, customDataUrl = null) {
    selectedFile = file;
    if (analyzeBtn) analyzeBtn.disabled = false;
    if (btnHelperText) btnHelperText.textContent = 'Ready for analysis execution';

    if (scanFilenameVal) scanFilenameVal.textContent = file.name || 'mri_brain_scan.png';
    if (scanFilesizeVal) {
      const kb = file.size ? Math.round(file.size / 1024) : 180;
      scanFilesizeVal.textContent = `${kb} KB`;
    }

    if (customDataUrl) {
      if (previewImg) previewImg.src = customDataUrl;
      if (dropzone) dropzone.style.display = 'none';
      if (filePreview) filePreview.style.display = 'flex';
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImg) previewImg.src = e.target.result;
      if (dropzone) dropzone.style.display = 'none';
      if (filePreview) filePreview.style.display = 'flex';
    };
    reader.readAsDataURL(file);
  }

  if (replaceFileBtn) {
    replaceFileBtn.addEventListener('click', () => {
      if (fileInput) fileInput.click();
    });
  }

  if (removeFileBtn) {
    removeFileBtn.addEventListener('click', () => {
      selectedFile = null;
      if (fileInput) fileInput.value = '';
      if (filePreview) filePreview.style.display = 'none';
      if (dropzone) dropzone.style.display = 'block';
      if (analyzeBtn) analyzeBtn.disabled = true;
      if (btnHelperText) btnHelperText.textContent = 'Select an MRI scan to begin.';
      sampleChips.forEach(c => c.classList.remove('selected'));
    });
  }

  // Refined Sample Scans
  const sampleChips = document.querySelectorAll('.sample-chip');
  sampleChips.forEach(chip => {
    chip.addEventListener('click', () => {
      sampleChips.forEach(c => c.classList.remove('selected'));
      chip.classList.add('selected');
      const type = chip.dataset.sample;
      generateSampleMriScan(type);
    });
  });

  function generateSampleMriScan(type) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#04070d';
    ctx.fillRect(0, 0, 256, 256);

    ctx.fillStyle = '#3a404d';
    ctx.beginPath();
    ctx.ellipse(128, 128, 92, 108, 0, 0, 2 * Math.PI);
    ctx.fill();

    ctx.fillStyle = '#5c6475';
    for (let i = 0; i < 8; i++) {
      ctx.beginPath();
      ctx.arc(90 + i * 10, 80 + (i % 3) * 30, 24, 0, Math.PI * 2);
      ctx.fill();
    }

    if (type !== 'normal') {
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      if (type === 'glioma') {
        ctx.ellipse(110, 100, 32, 24, Math.PI / 4, 0, Math.PI * 2);
      } else if (type === 'meningioma') {
        ctx.arc(175, 120, 22, 0, Math.PI * 2);
      } else if (type === 'pituitary') {
        ctx.ellipse(128, 182, 20, 15, 0, 0, Math.PI * 2);
      }
      ctx.fill();
    }

    const dataUrl = canvas.toDataURL('image/png');
    canvas.toBlob((blob) => {
      const file = new File([blob], `sample_${type}_mri.png`, { type: 'image/png' });
      handleFileSelected(file, dataUrl);
    }, 'image/png');
  }

  // Inspection Modal
  if (expandPreviewBtn) {
    expandPreviewBtn.addEventListener('click', () => {
      if (previewImg && previewImg.src) {
        if (inspectModalImg) inspectModalImg.src = previewImg.src;
        if (imageInspectModal) imageInspectModal.classList.add('open');
      }
    });
  }
  if (closeInspectModal) {
    closeInspectModal.addEventListener('click', () => {
      if (imageInspectModal) imageInspectModal.classList.remove('open');
    });
  }

  // Submit AI Analysis
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
      if (!selectedFile) return;

      analyzeBtn.disabled = true;
      if (analyzeBtnText) analyzeBtnText.textContent = 'Processing...';

      const formData = new FormData();
      formData.append('file', selectedFile);

      let endpoint = '/api/analyze/dual';
      if (currentMode === 'classify') endpoint = '/api/analyze/classify';
      if (currentMode === 'segment') endpoint = '/api/analyze/segment';

      try {
        const res = await fetch(endpoint, {
          method: 'POST',
          body: formData
        });

        if (!res.ok) {
          let errorMsg = `Server error (${res.status}). Please try again.`;
          try {
            const errData = await res.json();
            if (errData.error || errData.message) errorMsg = errData.error || errData.message;
          } catch (e) {
            // HTML error page (e.g. 502 Bad Gateway)
          }
          alert(errorMsg);
          return;
        }

        const data = await res.json();
        if (data.success) {
          displayResults(data);
        } else {
          alert(data.error || 'Failed to analyze MRI scan.');
        }
      } catch (err) {
        console.error('Analysis fetch error:', err);
        alert('Error connecting to ML backend server. Please try again.');
      } finally {
        analyzeBtn.disabled = false;
        updateAnalyzeBtnText();
      }
    });
  }

  function displayResults(data) {
    if (placeholderView) placeholderView.style.display = 'none';
    if (resultsSection) resultsSection.style.display = 'block';

    if (originalScanImg && data.original_image_url) {
      originalScanImg.src = data.original_image_url;
    }

    if (currentMode === 'classify') {
      if (maskStageBox) maskStageBox.style.display = 'none';
      if (overlayStageBox) overlayStageBox.style.display = 'none';
      if (opacityBarWrapper) opacityBarWrapper.style.display = 'none';
      if (tumorPctCard) tumorPctCard.style.display = 'none';
      if (tumorAreaCard) tumorAreaCard.style.display = 'none';
      if (classMetricCard) classMetricCard.style.display = 'block';
      if (confMetricCard) confMetricCard.style.display = 'block';
      if (probSection) probSection.style.display = 'block';
      if (modelUsedFootnote) modelUsedFootnote.textContent = 'VGG16 Transfer Learning Classifier';
    } else if (currentMode === 'segment') {
      if (maskStageBox) maskStageBox.style.display = 'flex';
      if (overlayStageBox) overlayStageBox.style.display = 'flex';
      if (opacityBarWrapper) opacityBarWrapper.style.display = 'block';
      if (tumorPctCard) tumorPctCard.style.display = 'block';
      if (tumorAreaCard) tumorAreaCard.style.display = 'block';
      if (classMetricCard) classMetricCard.style.display = 'none';
      if (confMetricCard) confMetricCard.style.display = 'none';
      if (probSection) probSection.style.display = 'none';
      if (modelUsedFootnote) modelUsedFootnote.textContent = 'U-Net Biomedical Encoder-Decoder Segmenter';
    } else {
      if (maskStageBox) maskStageBox.style.display = 'flex';
      if (overlayStageBox) overlayStageBox.style.display = 'flex';
      if (opacityBarWrapper) opacityBarWrapper.style.display = 'block';
      if (tumorPctCard) tumorPctCard.style.display = 'block';
      if (tumorAreaCard) tumorAreaCard.style.display = 'block';
      if (classMetricCard) classMetricCard.style.display = 'block';
      if (confMetricCard) confMetricCard.style.display = 'block';
      if (probSection) probSection.style.display = 'block';
      if (modelUsedFootnote) modelUsedFootnote.textContent = 'VGG16 Transfer Learning & U-Net Segmentation';
    }

    if (maskScanImg && data.mask_image_url) maskScanImg.src = data.mask_image_url;
    if (overlayScanImg && data.overlay_image_url) overlayScanImg.src = data.overlay_image_url;

    if (data.predicted_class) {
      if (classBadge) classBadge.textContent = data.predicted_class;
      if (confidenceBadge) confidenceBadge.textContent = `${data.confidence_score}%`;

      if (data.probabilities && probBarsContainer) {
        probBarsContainer.innerHTML = '';
        Object.entries(data.probabilities).forEach(([cls, pct]) => {
          const row = document.createElement('div');
          row.className = 'prob-row';
          row.innerHTML = `
            <div class="prob-meta">
              <span>${cls}</span>
              <span class="text-indigo" style="font-weight: 700;">${pct}%</span>
            </div>
            <div class="prob-track">
              <div class="prob-fill" style="width: ${Math.max(2, pct)}%;"></div>
            </div>
          `;
          probBarsContainer.appendChild(row);
        });
      }
    }

    if (data.tumor_percentage !== undefined) {
      if (tumorPctBadge) tumorPctBadge.textContent = `${data.tumor_percentage}%`;
      if (tumorAreaBadge) tumorAreaBadge.textContent = `${(data.tumor_area_pixels || 0).toLocaleString()} px`;
    }
  }

  if (opacitySlider) {
    opacitySlider.addEventListener('input', (e) => {
      const val = e.target.value;
      if (opacityValText) opacityValText.textContent = `${val}%`;
      if (overlayScanImg) overlayScanImg.style.opacity = val / 100;
    });
  }

  // Analytics Loader
  window.loadAnalytics = async function() {
    try {
      const res = await fetch('/api/analytics');
      const data = await res.json();
      if (data.success && data.analytics) {
        const a = data.analytics;
        const totalEl = document.getElementById('statTotalScans');
        const classEl = document.getElementById('statClassifications');
        const segEl = document.getElementById('statSegmentations');
        const confEl = document.getElementById('statAvgConfidence');

        if (totalEl) totalEl.textContent = a.total_scans || 0;
        if (classEl) classEl.textContent = a.total_classifications || 0;
        if (segEl) segEl.textContent = a.total_segmentations || 0;
        if (confEl) confEl.textContent = `${a.avg_confidence || 0}%`;

        const container = document.getElementById('analyticsClassBars');
        if (container && a.class_distribution) {
          container.innerHTML = '';
          const total = a.total_scans || 1;
          const entries = Object.entries(a.class_distribution);
          if (entries.length === 0) {
            container.innerHTML = `<p style="color: var(--text-muted); font-size: 0.85rem;">No scan classifications logged yet. Run predictions in Diagnostic Suite to populate category statistics.</p>`;
          } else {
            entries.forEach(([cls, count]) => {
              const pct = Math.round((count / total) * 100);
              const row = document.createElement('div');
              row.className = 'prob-row';
              row.style.marginBottom = '1rem';
              row.innerHTML = `
                <div class="prob-meta">
                  <span style="font-weight: 600;">${cls}</span>
                  <span class="text-indigo" style="font-weight: 700;">${count} scans (${pct}%)</span>
                </div>
                <div class="prob-track" style="height: 8px;">
                  <div class="prob-fill" style="width: ${Math.max(4, pct)}%;"></div>
                </div>
              `;
              container.appendChild(row);
            });
          }
        }
      }
    } catch (err) {
      console.error('Analytics fetch error:', err);
    }
  };

  // Audit Log Loader
  window.loadHistory = async function() {
    const tableBody = document.getElementById('historyTableBody');
    if (!tableBody) return;
    tableBody.innerHTML = `<tr><td colspan="8" class="table-empty-cell"><i class="fas fa-spinner fa-spin"></i> Loading radiology audit records...</td></tr>`;

    try {
      const res = await fetch('/api/history');
      const data = await res.json();
      if (data.success && data.records) {
        if (data.records.length === 0) {
          tableBody.innerHTML = `<tr><td colspan="8" class="table-empty-cell">No radiology audit records logged yet. Upload an MRI scan in Diagnostic Suite to populate log.</td></tr>`;
          return;
        }

        tableBody.innerHTML = '';
        data.records.forEach(r => {
          const tr = document.createElement('tr');
          const dateStr = r.upload_time ? new Date(r.upload_time).toLocaleString() : 'Recent';
          const confStr = r.confidence_score ? `${(r.confidence_score * 100).toFixed(1)}%` : '--';
          const pctStr = r.tumor_percentage !== null ? `${r.tumor_percentage}%` : '--';

          tr.innerHTML = `
            <td>#${r.id}</td>
            <td><i class="fas fa-user-md icon-blue"></i> ${r.username || 'Dr. Sapna'}</td>
            <td style="font-family: monospace; font-size: 0.825rem;">${r.filename || 'mri_scan.png'}</td>
            <td><span class="format-pill">${r.model_type || 'Dual'}</span></td>
            <td><strong class="text-indigo">${r.predicted_class || 'Segmentation'}</strong></td>
            <td>${confStr}</td>
            <td class="text-teal" style="font-weight: 600;">${pctStr}</td>
            <td style="font-size: 0.8rem; color: var(--text-muted);">${dateStr}</td>
          `;
          tableBody.appendChild(tr);
        });
      }
    } catch (err) {
      tableBody.innerHTML = `<tr><td colspan="8" class="table-empty-cell text-danger">Error loading audit log records.</td></tr>`;
    }
  };

  const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');
  if (refreshHistoryBtn) refreshHistoryBtn.addEventListener('click', loadHistory);

  // Initialize Auth Check on Boot
  checkSessionOnBoot();
});
