/**
 * Officer Authentication & Session Controller
 */

class OfficerAuth {
  static getToken() {
    return localStorage.getItem("officer_token");
  }

  static getProfile() {
    const raw = localStorage.getItem("officer_profile");
    return raw ? JSON.parse(raw) : null;
  }

  static setSession(token, user, profile) {
    localStorage.setItem("officer_token", token);
    localStorage.setItem("officer_user", JSON.stringify(user));
    localStorage.setItem("officer_profile", JSON.stringify(profile));
  }

  static logout() {
    localStorage.removeItem("officer_token");
    localStorage.removeItem("officer_user");
    localStorage.removeItem("officer_profile");
    window.location.href = "/officer/login";
  }

  static checkAuth() {
    const token = this.getToken();
    if (!token && !window.location.pathname.includes("/login") && !window.location.pathname.includes("/register")) {
      window.location.href = "/officer/login";
    }
  }
}

window.OfficerAuth = OfficerAuth;
