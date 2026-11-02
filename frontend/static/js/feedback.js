'use strict';

// Feedback page controller: validates ratings and sends them to the backend.
const feedbackPage = {
  currentMessageId: null,
  selectedRating: 0,
  selectedCorrectness: null,
  selectedLength: null,

  // Read the selected message id from the URL and prepare the form.
  init() {
    const params = new URLSearchParams(window.location.search);
    feedbackPage.currentMessageId = params.get('message_id');

    document.querySelectorAll('.star').forEach(star => {
      star.addEventListener('click', () => {
        feedbackPage.setRating(parseInt(star.dataset.value, 10));
      });
    });

    feedbackPage.bindOptionGroup('correctness-group', 'selectedCorrectness');
    feedbackPage.bindOptionGroup('length-group', 'selectedLength');
    document.getElementById('btn-submit-feedback').addEventListener('click', feedbackPage.submit);
  },

  // One-click option groups for correctness and response length.
  bindOptionGroup(groupId, fieldName) {
    document.querySelectorAll(`#${groupId} .toggle-btn`).forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll(`#${groupId} .toggle-btn`).forEach(item => item.classList.remove('selected'));
        btn.classList.add('selected');
        feedbackPage[fieldName] = btn.dataset.value;
      });
    });
  },

  setRating(value) {
    feedbackPage.selectedRating = value;
    document.querySelectorAll('.star').forEach(star => {
      star.classList.toggle('active', parseInt(star.dataset.value, 10) <= value);
    });
  },

  // Submit feedback, then return the user to the chat page.
  async submit() {
    const statusEl = document.getElementById('feedback-status');
    statusEl.textContent = '';
    statusEl.className = 'feedback-status';

    // Message-specific feedback needs every field; general bot feedback
    // (no message selected) only needs a rating or a comment.
    const comment = document.getElementById('feedback-comment').value.trim();
    if (feedbackPage.currentMessageId) {
      if (!feedbackPage.selectedRating) {
        feedbackPage.showError('Please select a star rating.');
        return;
      }
      if (!feedbackPage.selectedCorrectness) {
        feedbackPage.showError('Please select a correctness option.');
        return;
      }
      if (!feedbackPage.selectedLength) {
        feedbackPage.showError('Please select a length option.');
        return;
      }
    } else if (!feedbackPage.selectedRating && !comment) {
      feedbackPage.showError('Select a rating or write a comment.');
      return;
    }

    const payload = {
      message_id: feedbackPage.currentMessageId,
      rating: feedbackPage.selectedRating || null,
      correctness: feedbackPage.selectedCorrectness,
      length_rating: feedbackPage.selectedLength,
      comment: comment || null,
    };

    try {
      const res = await fetch('/feedback/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

      statusEl.textContent = 'Feedback saved. Returning to chat...';
      setTimeout(() => {
        window.location.href = '/chat';
      }, 600);
    } catch (err) {
      feedbackPage.showError(err.message);
    }
  },

  showError(message) {
    const statusEl = document.getElementById('feedback-status');
    statusEl.textContent = message;
    statusEl.className = 'feedback-status error';
  },
};

document.addEventListener('DOMContentLoaded', () => feedbackPage.init());
