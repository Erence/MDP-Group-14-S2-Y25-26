package com.example.mdp_14;

import android.content.Context;
import android.content.ContextWrapper;
import android.content.res.Configuration;
import android.content.res.Resources;
import java.util.Locale;

public class LocaleContextWrapper extends ContextWrapper {

    public LocaleContextWrapper(Context base) {
        super(base);
    }

    public static ContextWrapper wrap(Context context, Locale newLocale) {
        Resources res = context.getResources();
        Configuration config = new Configuration(res.getConfiguration());
        config.setLocale(newLocale);
        Context newContext = context.createConfigurationContext(config);
        return new ContextWrapper(newContext);
    }
}